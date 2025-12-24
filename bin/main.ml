(* Linear Gaussian Dynamics on HVS data with the new modular framework *)
open Base
open Ilqr_vae
open Misc
open Vae
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

let _ =
  Random.init 1998;
  Owl_stats_prng.init (Random.int 100000)


let in_dir = Cmdargs.in_dir "-results"
let data_path = Option.value_exn (Cmdargs.get_string "-data")

(* -----------------------------------------
   ---- Training Arguments -----
   ----------------------------------------- *)
let n_trials_save = 100
let n_samples = 100
let max_iter = 10000

(* state dim *)
let n = Option.value_exn (Cmdargs.get_int "-n")

(* control dim *)
let m = Option.value_exn (Cmdargs.get_int "-m")

(* whether to use statistics from data to initialise parameters *)
let a_init = Option.value (Cmdargs.get_bool "-a_init") ~default:false
let b_init = Option.value (Cmdargs.get_bool "-b_init") ~default:false
let noise_init = Option.value (Cmdargs.get_bool "-noise_init") ~default:false
let pin_prior = Option.value (Cmdargs.get_bool "-pin_prior") ~default:false

(* learning rate *)
let lr = Option.value (Cmdargs.get_float "-lr") ~default:0.001
let decay_rate = Option.value (Cmdargs.get_float "-decay_rate") ~default:1.
let mini_batch = Option.value (Cmdargs.get_int "-mini_batch") ~default:32
let early_stopping = Option.value (Cmdargs.get_bool "-early_stopping") ~default:false

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
type init_info =
  { c_init : AA.arr
  ; a_init : AA.arr option
  ; b_init : AA.arr option
  ; noise_init : AA.arr option
  }

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
let c_init pcs =
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


let z_0 z = AA.get_slice [ []; [ 0; -2 ]; [] ] z
let z_1 z = AA.get_slice [ []; [ 1; -1 ]; [] ] z

(* regression to fit state transition matrix. a = (z_1 z_0^T)^{-1}(z_0 z_0^T) *)
let compute_a ~z_0 ~z_1 =
  (* [ dim x (5000 x tmax-1) ] *)
  let z_0 = AA.transpose z_0
  and z_1 = AA.transpose z_1 in
  let to_inv = Mat.(z_0 *@ transpose z_0) in
  Mat.(z_1 *@ transpose z_0 *@ svd_inv to_inv)


(* std of state residual for scale of b matrix initialisation *)
let compute_b ~z_0 ~z_1 ~a =
  let a_z0 = Mat.(z_0 *@ transpose a) in
  (* [ 5000 x (tmax-1) x dim ] *)
  let diff = AA.(reshape (z_1 - a_z0) (shape z_0)) in
  AA.std ~axis:0 ~keep_dims:true diff


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


(* Compute info needed for initialising parameters *)
let param_init_info ~dim x =
  (* use a subset of x for computation *)
  let ids = List.range 0 AA.(shape x).(0) |> List.permute |> List.sub ~pos:0 ~len:5000 in
  let x = AA.get_fancy [ L ids; R []; R [] ] x in
  let pcs = pcs ~dim x in
  let z = z_inferred ~pcs x in
  let z_0 = z_0 z
  and z_1 = z_1 z in
  let z_0_ = AA.reshape z_0 [| -1; dim |] in
  let z_1_ = AA.reshape z_1 [| -1; dim |] in
  (* [ 5000 x (tmax-1) x dim ] *)
  let a_init =
    if a_init
    then (
      let a = compute_a ~z_0:z_0_ ~z_1:z_1_ in
      Mat.save_txt a ~out:(in_dir "a_init");
      Some a)
    else None
  in
  let b_init =
    if b_init
    then (
      let b =
        let a =
          match a_init with
          | None -> compute_a ~z_0:z_0_ ~z_1:z_1_
          | Some a -> a
        in
        compute_b ~z_0:z_0_ ~z_1:z_1_ ~a
      in
      Some b)
    else None
  in
  let noise_init =
    if noise_init
    then (
      let noise_init = compute_noise_init ~z ~c:pcs ~x in
      Some noise_init)
    else None
  in
  { a_init; b_init; c_init = c_init pcs; noise_init }


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
   -- Model Set-up ---
   ----------------------------------------- *)
type setup =
  { n : int
  ; m : int
  ; n_trials : int
  ; n_steps : int
  }

module Make_model (P : sig
    val setup : setup
    val n_beg : int Option.t
  end) =
struct
  module U = Prior.Gaussian (struct
      let n_beg = P.n_beg
    end)

  module UR = Prior.Gaussian (struct
      let n_beg = P.n_beg
    end)

  module L = Likelihood.Gaussian (struct
      let label = "o"
      let normalize_c = false
    end)

  module D = Dynamics.Linear_unconstrained (struct
      let n_beg = P.n_beg
    end)

  module Model =
    Vae.Make (U) (UR) (D) (L)
      (struct
        let n = P.setup.n
        let m = P.setup.m
        let n_steps = P.setup.n_steps
        let diag_time_cov = false
        let n_beg = P.n_beg
      end)
end

(* -----------------------------------------
   -- Initialise parameters and train
   ----------------------------------------- *)
(* sampling frequency of the data *)
let fs = 651.
let dt = Float.(1. / fs)
let tau = 0.02

(* observation dim *)
let n_output = 16 * 16
let setup = { n; m; n_trials = train_batch_size; n_steps = tmax }

module M = Make_model (struct
    let setup = setup
    let n_beg = Some (setup.n / setup.m)
  end)

open M

let reg ~(prms : Model.P.t') =
  let z = Float.(1e-5 / of_int Int.(setup.n * setup.n)) in
  let a = prms.dynamics.a in
  let a_reg = AD.Maths.(F z * l2norm_sqr' a) in
  match prms.dynamics.b with
  | None -> a_reg
  | Some b ->
    let b_reg = AD.Maths.(F z * l2norm_sqr' b) in
    AD.Maths.(a_reg + b_reg)


let init_prms () =
  C.broadcast' (fun () ->
    let n = setup.n
    and m = setup.m in
    (* pin prior and prior_recog *)
    let prior = U.init ~spatial_std:1.0 ~m () in
    let prior = if pin_prior then U.P.map prior ~f:Prms.pin else prior in
    let prior_recog = prior in
    let dynamics = D.init ~dt_over_tau:Float.(dt / tau) ~alpha:0.5 ~beta:0.5 ~n ~m () in
    let dynamics =
      D.P.
        { a =
            (match init_info.a_init with
             | None -> dynamics.a
             | Some a -> Prms.create (AD.pack_arr a))
        ; b =
            (match init_info.b_init with
             | None -> dynamics.b
             | Some b ->
               Some
                 (Prms.create
                    (AD.pack_arr Mat.(Float.(1. / of_int n) $* b * gaussian m n))))
        }
    in
    let likelihood =
      let likelihood_tmp =
        let tmp = L.init ~scale:1. ~sigma2:1. ~n ~n_output () in
        { tmp with c = Prms.create (AD.pack_arr init_info.c_init) }
      in
      { likelihood_tmp with
        variances =
          (match init_info.noise_init with
           | None -> likelihood_tmp.variances
           | Some noise_init -> Prms.create (AD.pack_arr noise_init))
      }
    in
    Model.init ~prior ~prior_recog ~dynamics ~likelihood ())


let save_results prefix prms data =
  let file s = prefix ^ "." ^ s in
  C.root_perform (fun () ->
    Misc.save_bin ~out:(file "params.bin") prms;
    Model.P.save_txt ~prefix prms);
  let prms = Model.P.value prms in
  (* sample from model parameters *)
  List.iter (List.range 0 n_trials_save) ~f:(fun i ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let data_gen = Model.sample_generative ~pre:true prms in
      let process_gen label a =
        let a = AD.unpack_arr a in
        AA.reshape a [| setup.n_steps; -1 |]
        |> AA.save_txt ~out:(file (Printf.sprintf "generated_%s_%i" label i))
      in
      process_gen "u" (Option.value_exn data_gen.u);
      process_gen "z" (Option.value_exn data_gen.z);
      process_gen "o" data_gen.o));
  (* sample from model using inferred u *)
  Array.iteri data ~f:(fun i dat_trial ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let mu = Model.posterior_mean ~prms dat_trial in
      AA.save_txt ~out:(file (Printf.sprintf "posterior_u_%i" i)) (AD.unpack_arr mu);
      let us, zs, os = Model.predictions ~n_samples ~prms mu in
      let process label a =
        let a = AD.unpack_arr a in
        AA.(mean ~axis:2 a @|| var ~axis:2 a)
        |> (fun z -> AA.reshape z [| setup.n_steps; -1 |])
        |> AA.save_txt ~out:(file (Printf.sprintf "predicted_%s_%i" label i))
      in
      process "u" us;
      process "z" zs;
      Array.iter ~f:(fun (label, x) -> process label x) os))


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


let t0 = Unix.gettimeofday ()

(* let rec iter ~stop ~k state =
  if stop
  then Optimizer.v state
  else (
    let prms = Model.broadcast_prms (Optimizer.v state) in
    if Int.(k % 100 = 0)
    then (
      let test_loss =
        Model.elbo_no_gradient
          ~n_samples
          ~conv_threshold:1E-4
          (Model.P.value prms)
          data_test
      in
      if C.first
      then (
        Optimizer.save ~out:(in_dir "state.bin") state;
        AA.(
          save_txt
            ~append:true
            ~out:(in_dir "test_loss")
            (of_array [| Float.of_int k; test_loss |] [| 1; 2 |]));
        save_results (in_dir "final") prms data_save_results));
    let loss, g =
      Model.elbo_gradient ~n_samples ~mini_batch ~conv_threshold:1E-4 ~reg prms data_train
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
    let t1 = Unix.gettimeofday () in
    let time_elapsed = t1 -. t0 in
    if Int.(k % 10 = 0)
    then print [%message (k : int) (time_elapsed : float) (loss : float)];
    if k < max_iter then iter ~stop ~k:(k + 1) state else Optimizer.v state) *)

let rec iter ~stop ~k ~prev_test_loss state =
  if stop
  then Optimizer.v state
  else (
    let prms = Model.broadcast_prms (Optimizer.v state) in
    (* Evaluate test loss every 100 iterations *)
    let stop, prev_test_loss =
      if Int.(k % 100 = 0)
      then (
        let test_loss =
          Model.elbo_no_gradient
            ~n_samples
            ~conv_threshold:1E-4
            (Model.P.value prms)
            data_test
        in
        (* Stop if test loss increased *)
        let stop =
          if early_stopping
          then (
            match prev_test_loss with
            | None -> false
            | Some prev -> Float.(test_loss > prev))
          else false
        in
        if C.first
        then (
          Optimizer.save ~out:(in_dir "state.bin") state;
          AA.(
            save_txt
              ~append:true
              ~out:(in_dir "test_loss")
              (of_array [| Float.of_int k; test_loss |] [| 1; 2 |]));
          save_results (in_dir "final") prms data_save_results);
        stop, Some test_loss)
      else stop, prev_test_loss
    in
    let loss, g =
      Model.elbo_gradient ~n_samples ~mini_batch ~conv_threshold:1E-4 ~reg prms data_train
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
    let t1 = Unix.gettimeofday () in
    let time_elapsed = t1 -. t0 in
    if Int.(k % 10 = 0)
    then print [%message (k : int) (time_elapsed : float) (loss : float)];
    let stop = Int.(k >= max_iter) || stop in
    iter ~stop ~k:(k + 1) ~prev_test_loss state)


let final_prms =
  let state =
    match Cmdargs.get_string "-reuse" with
    | Some file -> Optimizer.load file
    | None -> Optimizer.init (init_prms ())
  in
  iter ~stop:false ~k:0 ~prev_test_loss:None state


let _ = save_results (in_dir "final") final_prms data_save_results

(* compute validation loss from available models *)

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
