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


(* state dim *)
let n = Option.value_exn (Cmdargs.get_int "-n")

(* control dim *)
let m = Option.value_exn (Cmdargs.get_int "-m")
let in_dir = Cmdargs.in_dir "-results"
let n_trials_save = Cmdargs.get_int "-n_trials_save" |> Cmdargs.default 2
let n_steps = Cmdargs.get_int "-n_steps" |> Cmdargs.default 10
let chkpt_type = Cmdargs.get_string "-chkpt_type" |> Cmdargs.default "best"
let save_generative = Cmdargs.get_bool "-save_generative" |> Cmdargs.default true
let save_impulse = Cmdargs.get_bool "-save_impulse_response" |> Cmdargs.default false

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

open M

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
  let prepend = "generative" in
  (* sample from model parameters *)
  List.iter (List.range 0 n_trials_save) ~f:(fun i ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let u, z, o, o_noisy = Model.sample_generative ~noisy:true ~prms in
      process_gen ~i ~prefix ~prepend "u" u;
      process_gen ~i ~prefix ~prepend "z" z;
      process_gen ~i ~prefix ~prepend "o" o;
      process_gen ~i ~prefix ~prepend "o_noise" (Option.value_exn o_noisy)))


let save_impulse_response ~prefix prms =
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


(* Simulate from model parameters *)
let _ =
  let (prms : Model.P.t') =
    C.broadcast' (fun () ->
      Misc.read_bin (in_dir chkpt_type ^ ".params.bin") |> Model.P.value)
  in
  if save_generative
  then
    save_generative_results
      ~prefix:(in_dir chkpt_type ^ "_t_" ^ Int.to_string n_steps)
      prms;
  if save_impulse
  then
    save_impulse_response ~prefix:(in_dir chkpt_type ^ "_t_" ^ Int.to_string n_steps) prms
