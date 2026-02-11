open Base
open Ilqr_vae
open Misc
open Config

type setup =
  { n : int
  ; nh : int
  ; m : int
  ; n_steps : int
  ; dt : float
  }

let default_setup n_steps = { n = 32; nh = 128; m = 32; dt; n_steps }

module Make_model (P : sig
    val setup : setup
  end) =
struct
  module U = Prior.Student

  module L = Likelihood.Gaussian (struct
      let label = "o"
      let normalize_c = false
    end)

  module D = Dynamics.GNODE (struct
      let phi = AD.requad, AD.d_requad
    end)

  module Model =
    Vae.Make (U) (U) (D) (L)
      (struct
        let n = P.setup.n
        let m = P.setup.m
        let n_steps = P.setup.n_steps
        let diag_time_cov = false
      end)

  let init ~n_neurons ~c =
    let n = P.setup.n
    and nh = P.setup.nh
    and m = P.setup.m in
    let prior_recog = U.init ~spatial_std:1.0 ~m () in
    let dynamics = D.init ~radius:1. ~dt:P.setup.dt ~tau:0.1 ~n ~nh () in
    let likelihood =
      let likelihood = L.init ~n:P.setup.n ~n_output:n_neurons () in
      { likelihood with c = Prms.free AD.Maths.(F 0.1 * AD.pack_arr c) }
    in
    Model.init ~prior_recog ~dynamics ~likelihood ()
end
