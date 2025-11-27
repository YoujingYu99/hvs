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
let data_folder = Option.value_exn (Cmdargs.get_string "-data")

(* -----------------------------------------
   -- Data Read In ---
   ----------------------------------------- *)

let load_npy_data data_folder =
  (* get list of .npy files, each contains a mat of shape [T x n_channels]. n_channels
    are organized as [AP x ML]. *)
  let files = Stdlib.Sys.readdir data_folder |> List.of_array in
  let npy_files = List.filter ~f:(fun f -> Stdlib.Filename.check_suffix f ".npy") files in
  let full_paths =
    List.map ~f:(fun filename -> Stdlib.Filename.concat data_folder filename) npy_files
  in
  (* load data *)
  List.map ~f:(fun path -> Arr.load_npy path) full_paths


(* array of [T x n_channels] files. *)
let data_raw = load_npy_data data_folder

let standardize (mat : Mat.mat) : Mat.mat =
  let mean_ = Mat.mean ~axis:0 mat in
  let std_ = Mat.std ~axis:0 mat in
  Mat.(div (mat - mean_) std_)


let chunking ~tmax mat =
  let shape = Arr.shape mat in
  let t = shape.(0) in
  let num_blocks = t / tmax in
  List.init num_blocks ~f:(fun i ->
    Arr.get_slice [ [ tmax * i; (tmax * (i + 1)) - 1 ]; [] ] mat)


let tmax = 400

(* array of [tmax x n_channels] files *)
let data =
  let chunk_lst =
    List.map data_raw ~f:(fun mat ->
      (* TODO: standardise per time series *)
      let mat = standardize mat in
      let tmax_mat = Mat.row_num mat in
      if tmax_mat < tmax then None else Some (chunking ~tmax mat))
  in
  List.concat
    (List.filter_map
       ~f:(function
         | Some lst -> Some lst
         | None -> None)
       chunk_lst)
  |> List.permute


let data_train, data_test =
  let full_batch_size = List.length data in
  List.split_n data Float.(to_int (of_int full_batch_size * 0.8))


(* 1760 if tmax=400 *)
let data_train_size = List.length data_train

(* array of length [n_trials], each o has shape [tmax x n_channels] *)
let sample_data bs () =
  if bs > 0
  then (
    let indices =
      List.permute (List.range 0 data_train_size) |> List.sub ~pos:0 ~len:bs
    in
    let bs_of_data = List.map indices ~f:(fun idx -> List.nth_exn data idx) in
    bs_of_data |> List.to_array)
  else data_test |> List.to_array


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
  module U = Prior.Student (struct
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
let max_iter = 200000

(* sampling frequency of the data *)
let fs = 651.
let dt = Float.(1. / fs)
let tau = 0.02

(* state dim *)
let n = Option.value_exn (Cmdargs.get_int "-n")

(* control dim *)
let m = Option.value_exn (Cmdargs.get_int "-m")

(* observation dim *)
let n_output = 16 * 16
let setup = { n; m; n_trials = 512; n_steps = tmax }

module M = Make_model (struct
    let setup = setup
    let n_beg = Some (setup.n / setup.m)
  end)

open M

let data =
  let o = sample_data setup.n_trials () in
  Array.map o ~f:(fun o -> { u = None; z = None; o = AD.pack_arr o })


let data_test =
  let o = sample_data Int.(neg 1) () in
  Array.map o ~f:(fun o -> { u = None; z = None; o = AD.pack_arr o })


let init_prms () =
  C.broadcast' (fun () ->
    let n = setup.n
    and m = setup.m in
    (* let prior = U.init ~spatial_std:1.0 ~nu:20. ~m () in *)
    let prior = U.init ~spatial_std:1.0 ~m () in
    let prior_recog = UR.init ~spatial_std:1.0 ~m () in
    let dynamics = D.init ~dt_over_tau:Float.(dt / tau) ~alpha:0.9 ~beta:0.1 ~n ~m () in
    let likelihood = L.init ~scale:1. ~sigma2:1. ~n ~n_output () in
    Model.init ~prior ~prior_recog ~dynamics ~likelihood ())


let save_results prefix prms data =
  let prms = C.broadcast prms in
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


module Optimizer = Opt.Shampoo.Make (Model.P)

let config _k = Opt.Shampoo.{ beta = 0.95; learning_rate = Some 0.1 }

let rec iter ~k state =
  if k % 200 = 0 then Optimizer.save ~out:(in_dir "state.bin") state;
  let prms = C.broadcast (Optimizer.v state) in
  (* if Int.(k % 200 = 0)
  then (
    let test_loss, _ =
      Model.elbo_gradient ~n_samples:100 ~conv_threshold:1E-4 prms data_test
    in
    (if C.first
     then
       AA.(
         save_txt
           ~append:true
           ~out:(in_dir "test_loss")
           (of_array [| test_loss |] [| 1; 1 |])));
    save_results (in_dir "final") prms data); *)
  let loss, g =
    Model.elbo_gradient ~n_samples:100 ~mini_batch:32 ~conv_threshold:1E-4 prms data
  in
  (if C.first
   then AA.(save_txt ~append:true ~out:(in_dir "loss") (of_array [| loss |] [| 1; 1 |])));
  print [%message (k : int) (loss : float)];
  let state =
    match g with
    | None -> state
    | Some g -> Optimizer.step ~config:(config k) ~info:g state
  in
  if k < max_iter then iter ~k:(k + 1) state else Optimizer.v state


let final_prms =
  let state =
    match Cmdargs.get_string "-reuse" with
    | Some file -> Optimizer.load file
    | None -> Optimizer.init ~epsilon:1e-4 (init_prms ())
  in
  iter ~k:0 state


let _ = save_results (in_dir "final") final_prms data
