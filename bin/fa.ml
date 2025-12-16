(* Factor analysis on HVS data *)
open Printf
open Base
open Owl
open Torch
open Forward_torch
open Sofo
module Arr = Dense.Ndarray.S
module Mat = Dense.Matrix.S

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
let mini_batch = 32

(* -----------------------------------------
   -- Data Read In ---
   ----------------------------------------- *)

(* Load + preprocess only on rank 0 *)
let data_train, data_test, tmax, train_batch_size =
  (* shape [n_trials x tmax x n_channels ]*)
  let data_train_npy = Arr.load_npy (data_path ^ "train_std.npy") in
  let data_test_npy = Arr.load_npy (data_path ^ "test_std.npy") in
  let data_train_shape = Arr.shape data_train_npy in
  (* let full_batch_size = data_shape.(0) in *)
  let train_batch_size = data_train_shape.(0)
  and tmax = data_train_shape.(1) in
  print
    [%message
      (Arr.shape data_train_npy : int array) (Arr.shape data_test_npy : int array)];
  data_train_npy, data_test_npy, tmax, train_batch_size


let base =
  Optimizer.Config.Base.
    { device = Torch.Device.cuda_if_available ()
    ; kind = Torch_core.Kind.(T f64)
    ; ba_kind = Bigarray.float64
    }


(* state dim *)
let n = Option.value_exn (Cmdargs.get_int "-n")
let o = 256
let bs = 32
let epoch_of t = Float.(of_int t * of_int bs / of_int train_batch_size)
let ones_o = Maths.(of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ o ]))
let id_o = Maths.(of_tensor (Tensor.eye ~n:o ~options:(base.kind, base.device)))
let ones_z = Maths.(of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ n ]))

(* Get top principal components *)
let c, c_inv =
  let x = data_train in
  let x = Arr.(reshape x [| -1; (shape x).(2) |]) in
  let x = Arr.(x - mean ~axis:0 x) in
  let x = Arr.transpose x in
  let ids = List.range 0 Arr.(shape x).(1) |> List.permute |> List.sub ~pos:0 ~len:5000 in
  let x = Arr.get_fancy [ R []; L ids ] x in
  (* do an SVD *)
  let u, _, _ = Owl.Linalg.S.svd x in
  let c = Arr.get_slice [ []; [ 0; n - 1 ] ] u in
  let c_inv =
    let u, s, vh = Owl.Linalg.S.svd u in
    Mat.(transpose vh / s *@ transpose u)
  in
  Maths.of_bigarray ~device:base.device c, Maths.of_bigarray ~device:base.device c_inv


let z_train, z_test =
  let data_train_t = Maths.of_bigarray ~device:base.device data_train in
  let data_test_t = Maths.of_bigarray ~device:base.device data_train in
  ( Maths.einsum [ c, "ij"; data_train_t, "klj" ] "kli"
  , Maths.einsum [ c, "ij"; data_test_t, "klj" ] "kli" )


let sample_data batch_size =
  let z_train_bs = Maths.slice ~dim:0 ~start:0 ~end_:batch_size z_train in
  (* [bs x tmax-1 x o]*)
  let train_data_0 = Maths.slice ~dim:1 ~start:0 ~end_:Int.(tmax - 1) z_train_bs in
  let train_data_1 = Maths.slice ~dim:1 ~start:1 ~end_:Int.(tmax) z_train_bs in
  (* [bs * tmax-1 x o]*)
  Maths.reshape train_data_0 ~shape:[ -1; o ], Maths.reshape train_data_1 ~shape:[ -1; o ]


module PP = struct
  type 'a p =
    { c : 'a
    ; sigma_o_prms : 'a
    }
  [@@deriving prms]
end

module P = PP.Make (Prms.Single)

let solver a y =
  let ell = Maths.cholesky a in
  let ell_t = Maths.transpose ell in
  let _x = Maths.linsolve_triangular ~left:false ~upper:true ell_t y in
  Maths.linsolve_triangular ~left:false ~upper:false ell _x


let z_opt ~c ~sigma_o y =
  let a = Maths.(einsum [ c, "ji"; c, "jk" ] "ik") in
  let a = Maths.(a + diag_embed ~offset:0 ~dim1:0 ~dim2:1 (sqr sigma_o)) in
  let solution = solver a y in
  Maths.(einsum [ c, "ij"; solution, "mj" ] "mi")


let d_opt_inv ~c ~sigma_o =
  let sigma_o_inv_vec =
    Maths.(
      of_tensor (Tensor.ones ~device:base.device ~kind:base.kind [ n ]) / sqr sigma_o)
  in
  let cct = Maths.(einsum [ c, "ij"; c, "kj" ] "ik") in
  let a = Maths.(einsum [ cct, "ik"; sigma_o_inv_vec, "k" ] "ik") in
  let d_opt_inv =
    Maths.(of_tensor (Tensor.eye ~n ~options:(base.kind, base.device)) + a)
  in
  d_opt_inv


let gaussian_llh ?mu ?(fisher_batched = false) ~std x =
  let inv_std = Maths.(f 1. / std) in
  let error_term =
    if fisher_batched
    then (
      (* batch dimension l is number of fisher samples *)
      let error =
        match mu with
        | None -> Maths.(einsum [ x, "lma"; inv_std, "a" ] "lma")
        | Some mu -> Maths.(einsum [ x - mu, "lma"; inv_std, "a" ] "lma")
      in
      Maths.einsum [ error, "lma"; error, "lma" ] "lm")
    else (
      let error =
        match mu with
        | None -> Maths.(einsum [ x, "ma"; inv_std, "a" ] "ma")
        | Some mu -> Maths.(einsum [ x - mu, "ma"; inv_std, "a" ] "ma")
      in
      Maths.einsum [ error, "ma"; error, "ma" ] "m")
  in
  let cov_term =
    let cov_term_shape = if fisher_batched then [ 1; 1 ] else [ 1 ] in
    Maths.(sum (log (sqr std))) |> Maths.reshape ~shape:cov_term_shape
  in
  let of_tensor_term =
    let o = x |> Maths.to_tensor |> Tensor.shape |> List.last_exn in
    Float.(log (2. * pi) * of_int o)
  in
  Maths.(0.5 $* (of_tensor_term $+ error_term + cov_term)) |> Maths.neg


let gaussian_llh_chol ?mu ~chol x =
  let ell_t = Maths.transpose chol in
  let error_term =
    let error =
      match mu with
      | None -> x
      | Some mu -> Maths.(x - mu)
    in
    let error = Maths.linsolve_triangular ~left:false ~upper:true ell_t error in
    Maths.einsum [ error, "ma"; error, "ma" ] "m"
  in
  let cov_term =
    Maths.(sum (log (sqr (diagonal ~offset:0 chol))) |> reshape ~shape:[ 1 ])
  in
  let of_tensor_term =
    let o = x |> Maths.to_tensor |> Tensor.shape |> List.last_exn in
    Float.(log (2. * pi) * of_int o)
  in
  Maths.(0.5 $* (of_tensor_term $+ error_term + cov_term)) |> Maths.neg


module M = struct
  module P = P

  let init : P.param =
    let c =
      Prms.Single.free (Maths.of_bigarray ~device:base.device (c_init ~dim:n data_train))
    in
    let sigma_o_prms =
      Prms.Single.bounded
        ~above:(Maths.f 0.0001)
        Maths.(zeros ~device:base.device ~kind:base.kind [ 1 ])
    in
    PP.{ c; sigma_o_prms }


  let kl ~z_sampled ~z_diff ~d_opt_chol =
    let prior_term = gaussian_llh ~std:ones_z z_sampled in
    let q_term = gaussian_llh_chol ~chol:d_opt_chol z_diff in
    Maths.(q_term - prior_term)


  let neg_elbo ~y_pred ~y ~z_sampled ~z_diff ~d_opt_chol (theta : _ Maths.some P.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let sigma_o_extended = Maths.(sigma_o * ones_o) in
    let llh = gaussian_llh ~mu:y_pred ~std:sigma_o_extended y in
    let kl = kl ~z_sampled ~z_diff ~d_opt_chol in
    Maths.(llh - kl) |> Maths.neg


  let d_opt_chol (theta : _ Maths.some P.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let d_opt_inv = d_opt_inv ~c:theta.c ~sigma_o in
    let d_opt_inv_chol = Maths.cholesky d_opt_inv in
    Maths.linsolve
      d_opt_inv_chol
      (Maths.eye n ~device:base.device ~kind:base.kind)
      ~left:true


  let f ~data:y (theta : _ Maths.some P.t) =
    (* sample u *)
    let d_opt_chol = d_opt_chol theta in
    let z_opt = z_opt ~c:theta.c ~sigma_o:Maths.(exp theta.sigma_o_prms * ones_o) y in
    let z_diff =
      let e = Maths.(any (randn_like z_opt)) in
      Maths.einsum [ e, "mj"; d_opt_chol, "ij" ] "mi"
    in
    let z_sampled =
      (* Maths.(const (primal_tensor_detach (z_opt + z_diff))) *)
      Maths.(z_opt + z_diff)
    in
    let y_pred = Maths.(z_sampled *@ theta.c) in
    let loss =
      let tmp = neg_elbo ~y_pred ~y ~z_sampled ~z_diff ~d_opt_chol theta in
      Maths.(tmp / f Float.(of_int o))
    in
    loss


  let marginal_cov_chol (theta : _ Maths.some P.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let sigma_o_extended = Maths.(sigma_o * ones_o) in
    Maths.(
      (transpose theta.c *@ theta.c)
      + diag_embed ~offset:0 ~dim1:0 ~dim2:1 (sqr sigma_o_extended))
    |> Maths.cholesky


  let neg_marginal_llh ~marginal_cov_chol ~y =
    gaussian_llh_chol ~chol:marginal_cov_chol y |> Maths.neg


  let d_opt_chol (theta : _ Maths.some P.t) =
    let sigma_o = Maths.exp theta.sigma_o_prms in
    let d_opt_inv = d_opt_inv ~c:theta.c ~sigma_o in
    let d_opt_inv_chol = Maths.cholesky d_opt_inv in
    Maths.linsolve
      d_opt_inv_chol
      (Maths.of_tensor (Tensor.eye ~n ~options:(base.kind, base.device)))
      ~left:true
end

module O = Optimizer.Adam (M.P)

let config =
  Optimizer.Config.Adam.
    { base
    ; beta_1 = 0.9
    ; beta_2 = 0.99
    ; eps = 1e-4
    ; learning_rate = Some 0.1
    ; weight_decay = None
    ; debias = false
    }


let train ~max_iter =
  let rec loop ~t ~state running_avg =
    Stdlib.Gc.major ();
    let data = sample_data bs in
    let theta = O.params state in
    let theta_ = O.P.value theta in
    let theta_dual =
      O.P.map theta_ ~f:(fun x ->
        let x =
          x |> Maths.to_tensor |> Tensor.copy |> Tensor.to_device ~device:base.device
        in
        let x = Tensor.set_requires_grad x ~r:true in
        Tensor.zero_grad x;
        Maths.of_tensor x)
    in
    let loss, true_g =
      let loss = M.f ~data (P.map theta_dual ~f:Maths.any) in
      let loss = Maths.to_tensor loss in
      Tensor.backward loss;
      ( Tensor.to_float0_exn loss
      , O.P.map2 theta (O.P.to_tensor theta_dual) ~f:(fun tagged p ->
          match tagged with
          | Prms.Pinned _ -> Maths.(f 0.)
          | _ -> Maths.of_tensor (Tensor.grad p)) )
    in
    let new_state = O.step ~config ~info:true_g state in
    let running_avg =
      let loss_avg =
        match running_avg with
        | [] -> loss
        | running_avg -> running_avg |> Array.of_list |> Owl.Stats.mean
      in
      if t % 10 = 0
      then (
        O.P.C.save
          (M.P.value (O.params state))
          ~kind:base.ba_kind
          ~out:(in_dir "adam_params");
        (* save loss & acc *)
        let e = epoch_of t in
        print [%message (e : float) (loss_avg : float)];
        Owl.Mat.(
          save_txt
            ~append:true
            ~out:(in_dir "loss")
            (of_array [| Float.of_int t; loss_avg |] 1 2)));
      []
    in
    if t < max_iter
    then loop ~t:Int.(t + 1) ~state:new_state (loss :: running_avg)
    else new_state
  in
  loop ~t:0 ~state:(O.init M.init) []


let _ = train ~max_iter
