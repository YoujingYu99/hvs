(* Linear Gaussian Dynamics on HVS data with the new modular framework *)
open Base
open Ilqr_vae
open Misc
open Vae
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
let n_trials_save = Cmdargs.get_int "-n_trials_save" |> Cmdargs.default 2
let n_steps = Cmdargs.get_int "-n_steps" |> Cmdargs.default 10
let chkpt_type = Cmdargs.get_string "-chkpt_type" |> Cmdargs.default "best"
let save_generative = Cmdargs.get_bool "-save_generative" |> Cmdargs.default true

let save_generative_autonomous =
  Cmdargs.get_bool "-save_generative_autonomous" |> Cmdargs.default false


let save_impulse = Cmdargs.get_bool "-save_impulse_response" |> Cmdargs.default false

let save_generative_autonomous_inferred =
  Cmdargs.get_bool "-save_generative_autonomous_inferred" |> Cmdargs.default false


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
let setup = { n; m; n_trials = n_trials_save; n_steps }
let n_beg = Int.(setup.n / setup.m)
let n_samples = 100

module M = Make_model (struct
    let setup = setup
    let n_beg = Some n_beg
  end)

(* open M *)

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
    let data_test_npy = Arr.load_npy (data_path ^ "train_std.npy") in
    let data_train_shape = Arr.shape data_test_npy in
    let data_n_steps = data_train_shape.(1) in
    print [%message (Arr.shape data_test_npy : int array)];
    let data_train = pack_data data_test_npy in
    let data_save_results =
      List.permute (List.range 0 data_train_shape.(0))
      |> List.sub ~pos:0 ~len:n_trials_save
      |> List.map ~f:(fun i -> data_train.(i))
      |> List.to_array
    in
    data_save_results, data_n_steps)


let file ~prefix s = prefix ^ "." ^ s

let process_gen ~i ~prefix ~prepend label a =
  let a = AD.unpack_arr a in
  let shape =
    if String.(label = "u")
    then [| setup.n_steps + n_beg - 1; -1 |]
    else [| setup.n_steps; -1 |]
  in
  AA.reshape a shape
  |> AA.save_txt ~out:(file ~prefix (Printf.sprintf "%s_%s_%i" prepend label i))


let save_generative_results ~prefix prms =
  let prepend = "generated" in
  let open M in
  (* sample from model parameters *)
  List.iter (List.range 0 n_trials_save) ~f:(fun i ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let u, z, o, o_noisy = Model.sample_generative ~noisy:true ~prms in
      process_gen ~i ~prefix ~prepend "u" u;
      process_gen ~i ~prefix ~prepend "z" z;
      process_gen ~i ~prefix ~prepend "o" o;
      process_gen ~i ~prefix ~prepend "o_noise" (Option.value_exn o_noisy)))


let save_autonomous_generative_results ~prefix prms =
  let prepend = "generated_autonomous" in
  let open M in
  (* sample from model parameters *)
  List.iter (List.range 0 n_trials_save) ~f:(fun i ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let u, z, o, o_noisy =
        Model.sample_generative_autonomous ~noisy:true ~sigma:0.1 ~prms ()
      in
      process_gen ~i ~prefix ~prepend "u" u;
      process_gen ~i ~prefix ~prepend "z" z;
      process_gen ~i ~prefix ~prepend "o" o;
      process_gen ~i ~prefix ~prepend "o_noise" (Option.value_exn o_noisy)))


let save_impulse_response ~prefix prms =
  let open M in
  let prepend = Printf.sprintf "impulse_channel" in
  let n_sim = Int.(setup.n_steps + n_beg - 1) in
  let u_channel =
    let impulse = AD.(Mat.ones n_beg 1) in
    let zeros = AD.(Mat.zeros (setup.n_steps - 1) 1) in
    AD.Maths.concatenate ~axis:0 [| impulse; zeros |]
  in
  List.iter (List.range 0 setup.m) ~f:(fun i ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let u =
        if i = 0
        then
          AD.Maths.concatenate ~axis:1 [| u_channel; AD.Mat.zeros n_sim (setup.m - 1) |]
        else if i = setup.m - 1
        then
          AD.Maths.concatenate ~axis:1 [| AD.Mat.zeros n_sim (setup.m - 1); u_channel |]
        else
          AD.Maths.concatenate
            ~axis:1
            [| AD.Mat.zeros n_sim i; u_channel; AD.Mat.zeros n_sim (setup.m - 1 - i) |]
      in
      let u_impulse, z_impulse, o_impulse, o_noisy_impulse =
        Model.sample_forward ~noisy:true ~u ~prms
      in
      process_gen ~i ~prefix ~prepend "u" u_impulse;
      process_gen ~i ~prefix ~prepend "z" z_impulse;
      process_gen ~i ~prefix ~prepend "o" o_impulse;
      process_gen ~i ~prefix ~prepend "o_noise" (Option.value_exn o_noisy_impulse)))


let save_autonomous_test_ic_results ~prefix prms data =
  let setup = { n; m; n_trials = n_trials_save; n_steps = data_n_steps } in
  let module M_D =
    Make_model (struct
      let setup = setup
      let n_beg = Some n_beg
    end)
  in
  let open M_D in
  (* sample from model using inferred u *)
  Array.iteri data ~f:(fun i dat_trial ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let mu = Model.posterior_mean ~prms dat_trial in
      AA.save_txt
        ~out:(file ~prefix (Printf.sprintf "posterior_u_%i" i))
        (AD.unpack_arr mu);
      let u_inits, us, zs, os = Model.predictions ~n_samples ~prms mu in
      let process ?(shape = [| data_n_steps; -1 |]) label a =
        let a = AD.unpack_arr a in
        AA.(mean ~axis:2 a @|| var ~axis:2 a)
        |> (fun z -> AA.reshape z shape)
        |> AA.save_txt ~out:(file ~prefix (Printf.sprintf "predicted_%s_%i" label i))
      in
      process "u" us;
      process "u_inits" ~shape:[| n_beg; -1 |] us;
      process "z" zs;
      AA.save_txt
        ~out:(file ~prefix (Printf.sprintf "predicted_o_data_%i" i))
        (AD.unpack_arr dat_trial.o);
      assert (Array.length os = 1);
      Array.iter ~f:(fun (label, x) -> process label x) os;
      (* u_init for init cond then autonomous dynamics *)
      let us_init = u_inits |> AD.unpack_arr |> AA.mean ~axis:2 ~keep_dims:false in
      let u, z, o, o_noisy =
        Model.sample_generative_autonomous ~noisy:true ~u_init:us_init ~sigma:0.1 ~prms ()
      in
      let prepend = "predicted_auto" in
      process_gen ~i ~prefix ~prepend "u" u;
      process_gen ~i ~prefix ~prepend "z" z;
      process_gen ~i ~prefix ~prepend "o" o;
      process_gen ~i ~prefix ~prepend "o_noise" (Option.value_exn o_noisy)))


(* Simulate from model parameters *)
let _ =
  let open M in
  let (prms : Model.P.t') =
    C.broadcast' (fun () ->
      Misc.read_bin (in_dir chkpt_type ^ ".params.bin") |> Model.P.value)
  in
  if save_generative_autonomous
  then
    save_autonomous_generative_results
      ~prefix:(in_dir chkpt_type ^ "_t_" ^ Int.to_string n_steps)
      prms;
  if save_generative
  then
    save_generative_results
      ~prefix:(in_dir chkpt_type ^ "_t_" ^ Int.to_string n_steps)
      prms;
  if save_impulse
  then
    save_impulse_response ~prefix:(in_dir chkpt_type ^ "_t_" ^ Int.to_string n_steps) prms;
  if save_generative_autonomous_inferred
  then save_autonomous_test_ic_results ~prefix:(in_dir chkpt_type) prms data_save_results
