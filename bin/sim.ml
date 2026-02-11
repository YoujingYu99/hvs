(* Linear Gaussian Dynamics on HVS data with the new modular framework *)
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


let data_path = Option.value_exn (Cmdargs.get_string "-data")

(* state dim *)
let n = Option.value_exn (Cmdargs.get_int "-n")

(* control dim *)
let m = Option.value_exn (Cmdargs.get_int "-m")
let in_dir = Cmdargs.in_dir "-results"
let n_trials_save = Cmdargs.get_int "-n_trials_save" |> Cmdargs.default 100
let n_steps = Cmdargs.get_int "-n_steps" |> Cmdargs.default 100
let chkpt_type = Cmdargs.get_string "-chkpt_type" |> Cmdargs.default "best"
let save_generative = Cmdargs.get_bool "-save_generative" |> Cmdargs.default true

let save_generative_autonomous_inferred =
  Cmdargs.get_bool "-save_generative_autonomous_inferred" |> Cmdargs.default false


(* -----------------------------------------
   -- Initialise parameters and train
   ----------------------------------------- *)

let setup = { n; m; n_trials = n_trials_save; n_steps }
let n_beg = Int.(setup.n / setup.m)

module M = Make_model_MGU (struct
    let setup = setup
    let n_beg = Some n_beg
  end)

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


(* Load + preprocess only on rank 0 *)
let data_save_results, data_n_steps =
  C.broadcast' (fun () ->
    (* shape [n_trials x tmax x n_channels ]*)
    let data_test_npy = Arr.load_npy (data_path ^ "test_std.npy") in
    let data_test_shape = Arr.shape data_test_npy in
    let data_n_steps = data_test_shape.(1) in
    print [%message (Arr.shape data_test_npy : int array)];
    let data_test = pack_data data_test_npy in
    let data_save_results =
      List.permute (List.range 0 data_test_shape.(0))
      |> List.sub ~pos:0 ~len:n_trials_save
      |> List.map ~f:(fun i -> data_test.(i))
      |> List.to_array
    in
    data_save_results, data_n_steps)


let file ~prefix s = prefix ^ "." ^ s

let process_gen ~i ?(n_steps = setup.n_steps) ~prefix ~prepend label a =
  let a = AD.unpack_arr a in
  let shape = if String.(label = "u") then [| n_steps; -1 |] else [| n_steps; -1 |] in
  AA.reshape a shape
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


let ic_only mu =
  let open AD.Maths in
  let mu0 = get_slice [ [ 0 ] ] mu in
  let rest = AD.Mat.zeros Int.(AD.(shape mu).(0) - 1) AD.(shape mu).(1) in
  concat ~axis:0 mu0 rest


let save_autonomous_test_ic_results ~prefix prms data =
  let setup = { n; m; n_trials = n_trials_save; n_steps = data_n_steps } in
  let module M_D =
    Make_model_MGU (struct
      let setup = setup
      let n_beg = Some n_beg
    end)
  in
  let open M_D in
  let process ~id ~prefix label a =
    a
    |> AD.unpack_arr
    |> (fun z -> AA.reshape z [| setup.n_steps; -1 |])
    |> AA.save_txt ~out:(file ~prefix (Printf.sprintf "predicted_%s_%i" label id))
  in
  (* sample from model using inferred u *)
  Array.iteri data ~f:(fun i dat_trial ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let mu : AD.t = Model.posterior_mean ~prms dat_trial in
      let us, zs, os = Model.predictions_deterministic ~prms mu in
      let us0, zs0, os0 = Model.predictions_deterministic ~prms (ic_only mu) in
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


(* Simulate from model parameters *)
let _ =
  let open M in
  let (prms : Model.P.t') =
    C.broadcast' (fun () ->
      Misc.read_bin (in_dir chkpt_type ^ ".params.bin") |> Model.P.value)
  in
  if save_generative
  then
    save_generative_results
      ~prefix:(in_dir chkpt_type ^ "_t_" ^ Int.to_string n_steps)
      prms;
  if save_generative_autonomous_inferred
  then save_autonomous_test_ic_results ~prefix:(in_dir chkpt_type) prms data_save_results
