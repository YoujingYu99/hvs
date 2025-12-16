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
let max_iter = 10000

(* state dim *)
let n = Option.value_exn (Cmdargs.get_int "-n")

(* control dim *)
let m = Option.value_exn (Cmdargs.get_int "-m")
let mini_batch = 32

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


(* Use PCA for initialising the C matrix. *)
let c_init ~dim x =
  let x = AA.(reshape x [| -1; (shape x).(2) |]) in
  let x = AA.(x - mean ~axis:0 x) in
  let x = AA.transpose x in
  let ids = List.range 0 AA.(shape x).(1) |> List.permute |> List.sub ~pos:0 ~len:5000 in
  let x = AA.get_fancy [ R []; L ids ] x in
  (* do an SVD *)
  let u, _, _ = AA.Linalg.svd x in
  let c = AA.get_slice [ []; [ 0; dim - 1 ] ] u |> fun x -> AA.(x / l2norm ~axis:1 x) in
  let q, _ = AA.Linalg.qr (AA.gaussian [| dim; dim |]) in
  AA.(dot c q)


(* Load + preprocess only on rank 0 *)
let tmax, pcs, data_train, train_batch_size, data_save_results, data_test =
  C.broadcast' (fun () ->
    (* shape [n_trials x tmax x n_channels ]*)
    let data_train_npy = Arr.load_npy (data_path ^ "train_std.npy") in
    let data_test_npy = Arr.load_npy (data_path ^ "test_std.npy") in
    let data_train_shape = Arr.shape data_train_npy in
    (* let full_batch_size = data_shape.(0) in *)
    let train_batch_size = data_train_shape.(0)
    and tmax = data_train_shape.(1) in
    (* let train_batch_size = Float.(to_int (of_int full_batch_size *. 0.8)) in *)
    (* let data_train = Arr.get_slice [ [ 0; train_batch_size - 1 ]; []; [] ] data_full in *)
    (* let data_test =
      Arr.get_slice [ [ train_batch_size; full_batch_size - 1 ]; []; [] ] data_full
    in *)
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
    let pcs = c_init ~dim:n data_train_npy in
    print [%message "Finished computing PCs"];
    tmax, pcs, data_train, train_batch_size, data_save_results, data_test)


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

  module D = Dynamics.Linear (struct
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
  let a = D.unpack_a ~prms:prms.dynamics in
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
    (* let prior = U.init ~spatial_std:1.0 ~m () in *)
    let prior = U.P.map (U.init ~spatial_std:1.0 ~m ()) ~f:Prms.pin in
    let prior_recog = prior in
    (* initialise away from margin *)
    let dynamics = D.init ~dt_over_tau:Float.(dt / tau) ~alpha:0.5 ~beta:0.5 ~n ~m () in
    (* use PCA to initialise C matrix *)
    let likelihood =
      let tmp = L.init ~scale:1. ~sigma2:1. ~n ~n_output () in
      (* tmp *)
      { tmp with c = Prms.create (AD.pack_arr pcs) }
    in
    Model.init ~prior ~prior_recog ~dynamics ~likelihood ())


let save_results prefix prms data =
  let file s = prefix ^ "." ^ s in
  C.root_perform (fun () ->
    Misc.save_bin ~out:(file "params.bin") prms;
    Model.P.save_txt ~prefix prms);
  let prms = Model.P.value prms in
  Array.iteri data ~f:(fun i dat_trial ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let mu = Model.posterior_mean ~prms dat_trial in
      AA.save_txt ~out:(file (Printf.sprintf "posterior_u_%i" i)) (AD.unpack_arr mu);
      let us, zs, os = Model.predictions ~n_samples:100 ~prms mu in
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
    { learning_rate = Some 0.001
    ; epsilon = 1E-4
    ; beta1 = 0.9
    ; beta2 = 0.999
    ; weight_decay = None
    ; debias = true
    }


let t0 = Unix.gettimeofday ()

let rec iter ~k state =
  let prms = Model.broadcast_prms (Optimizer.v state) in
  if Int.(k % 100 = 0)
  then (
    let test_loss =
      Model.elbo_no_gradient
        ~n_samples:100
        ~conv_threshold:1E-4
        (Model.P.value prms)
        data_test
    in
    print [%message (k : int) (test_loss : float)];
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
    Model.elbo_gradient
      ~n_samples:100
      ~mini_batch
      ~conv_threshold:1E-4
      ~reg
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
  let t1 = Unix.gettimeofday () in
  let time_elapsed = t1 -. t0 in
  print [%message (k : int) (time_elapsed : float) (loss : float)];
  if k < max_iter then iter ~k:(k + 1) state else Optimizer.v state


let final_prms =
  let state =
    match Cmdargs.get_string "-reuse" with
    | Some file -> Optimizer.load file
    | None -> Optimizer.init (init_prms ())
  in
  iter ~k:0 state


let _ = save_results (in_dir "final") final_prms data_save_results

(* compute validation loss from available models *)

let _ =
  let (prms_final : Model.P.t') =
    C.broadcast' (fun () -> Misc.read_bin (in_dir "final.params.bin") |> Model.P.value)
  in
  let test_loss =
    Model.elbo_no_gradient ~n_samples:100 ~conv_threshold:1E-4 prms_final data_test
  in
  if C.first
  then
    AA.(
      save_txt
        ~append:true
        ~out:(in_dir "final_test_loss")
        (of_array [| test_loss |] [| 1; 1 |]))
