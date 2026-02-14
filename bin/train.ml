(* MGU2 on HVS data with the new modular framework *)
open Base
open Ilqr_vae
open Misc
open Vae
open Config
open Models
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let _ =
  Random.init 1998;
  Owl_stats_prng.init (1998 + C.rank)


let in_dir = Cmdargs.in_dir "-results"
let data_path = Option.value_exn (Cmdargs.get_string "-data")

(* -----------------------------------------
   ---- Training Arguments -----
   ----------------------------------------- *)
let n_trials_save = 10
let max_iter = 10000

(* state dim *)
let n = Option.value_exn (Cmdargs.get_int "-n")

(* control dim *)
let m = Option.value_exn (Cmdargs.get_int "-m")

(* whether to use statistics from data to initialise parameters *)
let a_init = Cmdargs.get_bool "-a_init" |> Cmdargs.default false
let c_init = Cmdargs.get_bool "-c_init" |> Cmdargs.default true
let noise_init = Cmdargs.get_bool "-noise_init" |> Cmdargs.default false
let pin_prior = Cmdargs.get_bool "-pin_prior" |> Cmdargs.default false
let pin_b = Cmdargs.get_bool "-pin_b" |> Cmdargs.default false

(* learning rate *)
let lr = Cmdargs.get_float "-lr" |> Cmdargs.default 0.001
let decay_rate = Cmdargs.get_float "-decay_rate" |> Cmdargs.default 1.
let mini_batch = Cmdargs.get_int "-mini_batch" |> Cmdargs.default 32
let use_early_stopping = Cmdargs.get_bool "-early_stopping" |> Cmdargs.default false
let regularise = Cmdargs.get_bool "-regularise" |> Cmdargs.default false
let k = Cmdargs.get_int "-k" |> Cmdargs.default 0
let reuse = Cmdargs.get_string "-reuse"

(* -----------------------------------------
   -- Data Read In ---
   ----------------------------------------- *)

let pack_o o = { Vae.u = None; z = None; o = AD.pack_arr o }

(* truncate such that length of [data] is a multiple of [mb]. *)
let truncate data mb =
  let n_total = Array.length data in
  let n_mb =
    let ratio = Float.of_int n_total /. Float.of_int mb in
    Float.round_down ratio |> Int.of_float
  in
  let keep = n_mb * mb in
  Array.sub data ~pos:0 ~len:keep


let pack_data x =
  let d0, tmax, o_dim = (Arr.shape x).(0), (Arr.shape x).(1), (Arr.shape x).(2) in
  Array.init d0 ~f:(fun i ->
    let s = Arr.get_slice [ [ i ]; []; [] ] x in
    pack_o (Arr.reshape s [| tmax; o_dim |]))


(* -----------------------------------------
   -- Param Init Quantities ---
   ----------------------------------------- *)

(* Compute pcs. x has shape [n_trials x tmax x o] *)
let pcs ~dim x =
  let x = AA.(reshape x [| -1; (shape x).(2) |]) in
  let x = AA.(x - mean ~axis:0 x) in
  let x = AA.transpose x in
  (* do an SVD *)
  let u, _, _ = AA.Linalg.svd x in
  (* pcs has shape [o x dim] *)
  AA.get_slice [ []; [ 0; dim - 1 ] ] u


(* Use PCA for initialising the C matrix. *)
let init_c pcs =
  let c = AA.(pcs / l2norm ~axis:1 pcs) in
  let dim = (AA.shape pcs).(1) in
  let q, _ = AA.Linalg.qr (AA.gaussian [| dim; dim |]) in
  AA.(dot c q)


let svd_inv a =
  let u, s, vh = AA.Linalg.svd a in
  Mat.(transpose vh / (1e-6 $+ s) *@ transpose u)


let z_inferred ~pcs x =
  let c_inv = svd_inv pcs in
  let x_shape = Arr.shape x in
  let n_trials = x_shape.(0)
  and tmax = x_shape.(1) in
  let x = AA.(reshape x [| -1; (shape x).(2) |]) in
  let z = Mat.(x *@ transpose c_inv) in
  let z = AA.reshape z [| n_trials; tmax; -1 |] in
  z


(* regression to fit state transition matrix. a = (z_1 z_0^T)^{-1}(z_0 z_0^T) *)
let compute_a ~z ~dim =
  (* [ dim x (5000 x tmax-1) ] *)
  let z_0 =
    AA.get_slice [ []; [ 0; -2 ]; [] ] z
    |> fun x -> AA.reshape x [| -1; dim |] |> AA.transpose
  and z_1 =
    AA.get_slice [ []; [ 1; -1 ]; [] ] z
    |> fun x -> AA.reshape x [| -1; dim |] |> AA.transpose
  in
  let to_inv = Mat.(z_0 *@ transpose z_0) in
  Mat.(z_1 *@ transpose z_0 *@ svd_inv to_inv)


(* std of observation residual for scale of noise initialisation *)
let compute_noise_init ~z ~c ~x =
  let dim = (Arr.shape z).(2) in
  let pred =
    let z_tmp = Arr.(reshape z [| -1; dim |]) in
    Mat.(z_tmp *@ transpose c)
  in
  let diff =
    let x_tmp = Arr.reshape x [| -1; (Arr.shape x).(2) |] in
    Mat.(x_tmp - pred)
  in
  AA.var ~axis:0 ~keep_dims:true diff


type init_info =
  { a_init : AA.arr option
  ; c_init : AA.arr option
  ; noise_init : AA.arr option
  }

(* Compute info needed for initialising parameters *)
let param_init_info ~dim x =
  (* use a subset of x for computation *)
  let ids = List.range 0 AA.(shape x).(0) |> List.permute |> List.sub ~pos:0 ~len:1000 in
  let x = AA.get_fancy [ L ids; R []; R [] ] x in
  let pcs = pcs ~dim x in
  let z = z_inferred ~pcs x in
  (* [ 5000 x (tmax-1) x dim ] *)
  let a =
    if a_init
    then (
      let a = compute_a ~z ~dim in
      Some a)
    else None
  in
  let noise_init =
    if noise_init
    then (
      let noise_init = compute_noise_init ~z ~c:pcs ~x in
      Some noise_init)
    else None
  in
  let c_init = if c_init then Some (init_c pcs) else None in
  { a_init = a; c_init; noise_init }


(* Load + preprocess only on rank 0 *)
let tmax, init_info, data_train, train_batch_size, data_save_results, data_test =
  C.broadcast' (fun () ->
    (* shape [n_trials x tmax x n_channels ]*)
    let data_train_npy = Arr.load_npy (data_path ^ "train_std.npy") in
    let data_test_npy = Arr.load_npy (data_path ^ "test_std.npy") in
    let data_train_shape = Arr.shape data_train_npy in
    let train_batch_size = data_train_shape.(0)
    and tmax = data_train_shape.(1) in
    print
      [%message
        (Arr.shape data_train_npy : int array) (Arr.shape data_test_npy : int array)];
    let data_train = pack_data data_train_npy in
    let data_test = pack_data data_test_npy in
    let data_save_results =
      List.permute (List.range 0 data_train_shape.(0))
      |> List.sub ~pos:0 ~len:n_trials_save
      |> List.map ~f:(fun i -> data_train.(i))
      |> List.to_array
    in
    let data_test = truncate data_test mini_batch in
    let init_info = param_init_info ~dim:n data_train_npy in
    print [%message "Finished computing param init info"];
    tmax, init_info, data_train, train_batch_size, data_save_results, data_test)


(* -----------------------------------------
   -- Initialise parameters and train
   ----------------------------------------- *)
let setup = { n; m; n_steps = tmax; dt; nh = 128 }
let n_beg = Int.(setup.n / setup.m)

module M = Make_model_GNODE (struct
    let setup = setup
    let n_beg = Some n_beg
  end)

open M

(* if mgu *)
(* let reg ~(prms : Model.P.t') =
  let z = Float.(1e-5 / of_int Int.(setup.n * setup.n)) in
  (* if MGU *)
  let part1 = AD.Maths.(F z * l2norm_sqr' prms.dynamics.uh) in
  let part2 = AD.Maths.(F z * l2norm_sqr' prms.dynamics.uf) in
  AD.Maths.(part1 + part2) *)

(* if LDS *)
(* let a = prms.dynamics.a in
  let a_reg = AD.Maths.(F z * l2norm_sqr' a) in
  match prms.dynamics.b with
  | None -> a_reg
  | Some b ->
    let b_reg = AD.Maths.(F z * l2norm_sqr' b) in
    AD.Maths.(a_reg + b_reg) *)

(* if GNODE *)
let reg ~(prms : Model.P.t') =
  let z = Float.(1e-5 / of_int Int.(setup.n * setup.n)) in
  (* if MGU *)
  let part1 = AD.Maths.(F z * l2norm_sqr' prms.dynamics.g_a1) in
  let part2 = AD.Maths.(F z * l2norm_sqr' prms.dynamics.h_a1) in
  let part3 = AD.Maths.(F z * l2norm_sqr' prms.dynamics.h_a2) in
  AD.Maths.(part1 + part2 + part3)


open M

let reg_arg = if regularise then Some reg else None

let init_prms () =
  C.broadcast' (fun () ->
    let n = setup.n
    and m = setup.m in
    (* optionally pin prior and prior_recog *)
    let prior =
      let tmp = U.init ~spatial_std:1.0 ~m () in
      if pin_prior then U.P.map tmp ~f:Prms.pin else tmp
    in
    let prior_recog = UR.init ~spatial_std:1.0 ~m () in
    (* if LDS *)
    (* let dynamics = D.init ~dt_over_tau:Float.(dt / tau) ~alpha:0.5 ~beta:0.5 ~n ~m () in
    let dynamics =
      let a =
        match init_info.a_init with
        | None -> dynamics.a
        | Some a -> Prms.create (AD.pack_arr a)
      in
      let b = dynamics.b in
      let b = if pin_b then Option.map ~f:Prms.pin b else b in
      D.P.{ a; b }
    in *)
    (* if MGU *)
    (* let dynamics = D.init ~n ~m () in *)
    (* if GNODE *)
    let dynamics = D.init ~n ~m ~nh:setup.nh ~dt ~tau () in
    let likelihood =
      let likelihood_tmp =
        let tmp = L.init ~scale:1. ~sigma2:1. ~n ~n_output () in
        let c =
          match init_info.c_init with
          | None -> tmp.c
          | Some c -> Prms.create (AD.pack_arr c)
        in
        { tmp with c }
      in
      { likelihood_tmp with
        variances =
          (match init_info.noise_init with
           | None -> likelihood_tmp.variances
           | Some noise_init -> Prms.create (AD.pack_arr noise_init))
      }
    in
    Model.init ~prior ~prior_recog ~dynamics ~likelihood ())


let file ~prefix s = prefix ^ "." ^ s

let save_params ~prefix prms =
  C.root_perform (fun () ->
    Misc.save_bin ~out:(file ~prefix "params.bin") prms;
    Model.P.save_txt ~prefix prms)


let process_gen ~i ~prefix ~prepend label a =
  let a = AD.unpack_arr a in
  let n_steps = (AA.shape a).(0) in
  AA.reshape a [| n_steps; -1 |]
  |> AA.save_txt ~out:(file ~prefix (Printf.sprintf "%s_%s_%i" prepend label i))


let save_generative_results ~prefix prms =
  let prepend = "generated" in
  let open M in
  (* sample from model parameters *)
  List.iter (List.range 0 n_trials_save) ~f:(fun i ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let u, z, o, _ = Model.sample_generative ~noisy:false ~prms in
      process_gen ~i ~prefix ~prepend "u" u;
      process_gen ~i ~prefix ~prepend "z" z;
      process_gen ~i ~prefix ~prepend "o" o))


let ic_only ~n_steps mu =
  let open AD.Maths in
  let mu0 = get_slice [ [ 0 ] ] mu in
  let rest = AD.Mat.zeros Int.(n_steps - 1) AD.(shape mu).(1) in
  concat ~axis:0 mu0 rest


let save_autonomous_test_ic_results ~prefix prms data =
  let process ~id ~prefix label a =
    a
    |> AD.unpack_arr
    |> (fun z -> AA.reshape z [| AA.(shape (AD.unpack_arr a)).(0); -1 |])
    |> AA.save_txt ~out:(file ~prefix (Printf.sprintf "predicted_%s_%i" label id))
  in
  (* sample from model using inferred u *)
  Array.iteri data ~f:(fun i dat_trial ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let mu : AD.t = Model.posterior_mean ~prms dat_trial in
      let us, zs, os = Model.predictions_deterministic ~prms mu in
      let us0, zs0, os0 =
        Model.predictions_deterministic ~prms (ic_only ~n_steps:setup.n_steps mu)
      in
      AA.save_txt
        ~out:(file ~prefix (Printf.sprintf "posterior_u_%i" i))
        (AD.unpack_arr mu);
      (* model inferred u *)
      process ~id:i ~prefix "u" us;
      process ~id:i ~prefix "z" zs;
      process ~id:i ~prefix "o" (snd os.(0));
      (* autonomous ic *)
      process ~id:i ~prefix "ic_u" us0;
      process ~id:i ~prefix "ic_z" zs0;
      process ~id:i ~prefix "ic_o" (snd os0.(0))))


let save_results ~prefix prms data =
  save_params ~prefix prms;
  let prms = Model.P.value prms in
  (* sample from model parameters *)
  save_generative_results ~prefix prms;
  save_autonomous_test_ic_results ~prefix prms data


module Optimizer = Opt.Adam.Make (Model.P)

let config _k =
  Opt.Adam.
    { learning_rate = Some Float.(lr / (1. + (decay_rate * sqrt (of_int _k))))
    ; epsilon = 1E-4
    ; beta1 = 0.9
    ; beta2 = 0.999
    ; weight_decay = None
    ; debias = true
    }


type early_stop_state =
  { best_loss : float option
  ; wait : int
  ; patience : int
  }

let check_early_stop ~state current_loss =
  match state.best_loss with
  | None -> false, Some { state with best_loss = Some current_loss; wait = 0 }
  | Some best ->
    if Float.(current_loss < best)
    then false, Some { state with best_loss = Some current_loss; wait = 0 }
    else (
      let wait = state.wait + 1 in
      wait > state.patience, Some { state with wait })


let rec iter ~k ~state_early_stop ~curr_best state =
  let prms = Model.broadcast_prms (Optimizer.v state) in
  (* Evaluate test loss every 100 iterations *)
  let reached_best, curr_best, stop_test, state_early_stop =
    if Int.(k % 100 = 0)
    then (
      let test_loss =
        Model.elbo_no_gradient
          ~n_samples
          ~conv_threshold:1E-4
          (Model.P.value prms)
          data_test
      in
      let reached_best, curr_best =
        (match curr_best with
         | None -> true, Some test_loss
         | Some curr_best ->
           if Float.(test_loss < curr_best)
           then true, Some test_loss
           else false, Some curr_best)
        |> C.broadcast
      in
      if C.first
      then (
        Optimizer.save ~out:(in_dir "state.bin") state;
        AA.(
          save_txt
            ~append:true
            ~out:(in_dir "test_loss")
            (of_array [| Float.of_int k; test_loss |] [| 1; 2 |])));
      save_results ~prefix:(in_dir "final") prms data_save_results;
      let stop_early, state_early_stop =
        match state_early_stop with
        | Some state_early_stop -> check_early_stop ~state:state_early_stop test_loss
        | None -> false, None
      in
      reached_best, curr_best, stop_early, state_early_stop)
    else false, curr_best, false, state_early_stop
  in
  if reached_best then save_results ~prefix:(in_dir "best") prms data_save_results;
  let local_stop = if C.rank = 0 then stop_test || Int.(k >= max_iter) else false in
  let stop = C.broadcast local_stop in
  if stop
  then Optimizer.v state
  else (
    let loss, g =
      Model.elbo_gradient
        ~n_samples
        ~mini_batch
        ~conv_threshold:1E-4
        ?reg:reg_arg
        prms
        data_train
    in
    (if C.first
     then
       AA.(
         save_txt
           ~append:true
           ~out:(in_dir "loss")
           (of_array [| Float.of_int k; loss |] [| 1; 2 |])));
    let state =
      match g with
      | None -> state
      | Some g -> Optimizer.step ~config:(config k) ~info:g state
    in
    if Int.(k % 100 = 0) then print [%message (k : int) (loss : float)];
    iter ~k:(k + 1) ~state_early_stop ~curr_best state)


let final_prms =
  let state =
    match reuse with
    | Some file ->
      print [%message "reusing last checkpoint"];
      Optimizer.load file
    | None ->
      let init_params = init_prms () in
      save_params ~prefix:(in_dir "init") init_params;
      Optimizer.init init_params
  in
  print [%message "training starts"];
  let state_early_stop =
    if use_early_stopping then Some { best_loss = None; wait = 0; patience = 5 } else None
  in
  iter ~k ~state_early_stop ~curr_best:None state


let _ = save_results ~prefix:(in_dir "final") final_prms data_save_results

(* Compute final validation loss from available models *)
let _ =
  let (prms_final : Model.P.t') =
    C.broadcast' (fun () -> Misc.read_bin (in_dir "final.params.bin") |> Model.P.value)
  in
  let test_loss =
    Model.elbo_no_gradient ~n_samples ~conv_threshold:1E-4 prms_final data_test
  in
  if C.first
  then
    AA.(
      save_txt
        ~append:true
        ~out:(in_dir "final_test_loss")
        (of_array [| test_loss |] [| 1; 1 |]))
