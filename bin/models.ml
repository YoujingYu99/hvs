open Base
open Ilqr_vae
open Misc
module Mat = Owl.Dense.Matrix.S
module Arr = Owl.Dense.Ndarray.S

(* -----------------------------------------
    -- Model Set-up ---
    ----------------------------------------- *)
type setup =
  { n : int
  ; m : int
  ; nh : int
  ; n_trials : int
  ; n_steps : int
  }

module Make_model (P : sig
    val setup : setup
  end) =
struct
  module U = Prior.Laplacian
  module UR = Prior.Gaussian

  module L = Likelihood.Gaussian (struct
      let label = "o"
      let normalize_c = false
    end)

  module D = Dynamics.GNODE (struct
      let phi = AD.Maths.tanh, fun x -> AD.Maths.(F 1. - sqr (tanh x))
      let dt = 0.01
    end)

  module Model =
    Vae.Make (U) (UR) (D) (L)
      (struct
        let n = P.setup.n
        let m = P.setup.m
        let n_steps = P.setup.n_steps
        let diag_time_cov = false
      end)
end
