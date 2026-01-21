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
let n_trials_save = Cmdargs.get_int "-n_trials_save" |> Cmdargs.default 10
let n_steps = Cmdargs.get_int "-n_steps" |> Cmdargs.default 1000
let chkpt_type = Cmdargs.get_string "-chkpt_type" |> Cmdargs.default "best"

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
let n_samples = 100

module M = Make_model (struct
    let setup = setup
    let n_beg = Some (setup.n / setup.m)
  end)

open M

let file ~prefix s = prefix ^ "." ^ s

let save_results ~prefix prms =
  (* sample from model parameters *)
  List.iter (List.range 0 n_trials_save) ~f:(fun i ->
    if Int.(i % C.n_nodes = C.rank)
    then (
      let u, z, o, o_noisy = Model.sample_generative ~noisy:true prms in
      let process_gen label a =
        let a = AD.unpack_arr a in
        AA.reshape a [| setup.n_steps; -1 |]
        |> AA.save_txt ~out:(file ~prefix (Printf.sprintf "generated_%s_%i" label i))
      in
      process_gen "u" u;
      process_gen "z" z;
      process_gen "o" o;
      process_gen "o_noise" (Option.value_exn o_noisy)))


(* Simulate from model parameters *)
let _ =
  let (prms : Model.P.t') =
    C.broadcast' (fun () ->
      Misc.read_bin (in_dir chkpt_type ^ ".params.bin") |> Model.P.value)
  in
  save_results ~prefix:(in_dir chkpt_type ^ "_t_" ^ Int.to_string n_steps) prms
