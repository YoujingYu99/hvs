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
  ; nh : int
  ; m : int
  ; n_steps : int
  ; dt : float
  }

module Make_model_LDS (P : sig
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

module Make_model_MGU (P : sig
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

  module D = Dynamics.MGU2 (struct
      let n_beg = P.n_beg
      let phi x = AD.Maths.(AD.requad x - F 1.)
      let d_phi = AD.d_requad
      let sigma x = AD.Maths.sigmoid x

      let d_sigma x =
        let tmp = AD.Maths.(exp (neg x)) in
        AD.Maths.(tmp / sqr (F 1. + tmp))
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

module Make_model_GNODE (P : sig
    val setup : setup
    val n_beg : int Option.t
  end) =
struct
  module U = Prior.Student (struct
      let n_beg = P.n_beg
    end)

  module UR = Prior.Student (struct
      let n_beg = P.n_beg
    end)

  module L = Likelihood.Gaussian (struct
      let label = "o"
      let normalize_c = false
    end)

  module D = Dynamics.GNODE (struct
      let phi = AD.requad, AD.d_requad
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

  let init ~n_neurons ~c =
    let n = P.setup.n
    and nh = P.setup.nh
    and m = P.setup.m in
    let prior_recog = U.init ~spatial_std:1.0 ~m () in
    let dynamics = D.init ~radius:1. ~dt:P.setup.dt ~tau:0.1 ~n ~m ~nh () in
    let likelihood =
      let likelihood = L.init ~n:P.setup.n ~n_output:n_neurons () in
      { likelihood with c = Prms.free AD.Maths.(F 0.1 * AD.pack_arr c) }
    in
    Model.init ~prior_recog ~dynamics ~likelihood ()
end
