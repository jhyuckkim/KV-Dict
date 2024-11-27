import torch

def innerp(x, y=None, out=None):
    if y is None:
        y = x
    if out is not None:
        out = out[:, None, None]  # Add space for two singleton dimensions.
    return torch.matmul(x[..., None, :], y[..., :, None], out=out)[..., 0, 0]

def omp_v0(X, y, XTX, n_nonzero_coefs=None, tol=None):
    # X has shape (B, m, n) Y has shape (B, b, m)
    B, b, _ = y.shape
    normr2_init = innerp(y) # (b) -> (B, b)
    normr2 = normr2_init.clone()
    projections = torch.bmm(X.transpose(2, 1), y.transpose(1, 2)).transpose(1, 2)
    sets = y.new_zeros(n_nonzero_coefs, B, b, dtype=torch.int64)

    F = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, b, 1, 1)
    a_F = y.new_zeros((n_nonzero_coefs, B, b, 1), dtype=y.dtype)

    D_mybest = y.new_empty(B, b, n_nonzero_coefs, XTX.shape[1])
    temp_F_k_k = y.new_ones((B, b, 1))

    if tol:
        result_lengths = sets.new_zeros((y.shape[0], y.shape[1]))
        result_solutions = y.new_zeros((y.shape[0], y.shape[1], n_nonzero_coefs, 1))
        finished_problems = sets.new_zeros((y.shape[0], y.shape[1]), dtype=torch.bool)
        tol = normr2_init * (tol * tol)

    for k in range(n_nonzero_coefs+(tol is not None)):
        # STOPPING CRITERIA
        if tol is not None:
            problems_done = normr2 <= tol 
            if k == n_nonzero_coefs:
                below_tol = problems_done.clone()
                problems_done[:, :] = True
            
            if problems_done.any():
                new_problems_done = problems_done & ~finished_problems
                finished_problems.logical_or_(problems_done)
                result_lengths[new_problems_done] = k
                result_solutions.view(-1, n_nonzero_coefs, 1)[new_problems_done.flatten(), :k] = \
                    F.view(-1, n_nonzero_coefs, n_nonzero_coefs)[new_problems_done.flatten(), :k, :k].permute(0, 2, 1) @ a_F.view(n_nonzero_coefs, -1, 1)[:k, new_problems_done.flatten()].permute(1, 0, 2)

                if problems_done.all():
                    if k == n_nonzero_coefs:
                        # there may be elements below tol
                        return sets.permute(1, 2, 0), result_solutions, normr2, normr2_init, result_lengths, ~below_tol
                    else:
                        return sets.permute(1, 2, 0), result_solutions, normr2, normr2_init, result_lengths, ~problems_done

        sets[k] = projections.abs().argmax(2)
        torch.gather(XTX, 1, sets[k, :, :, None].expand(-1, -1, XTX.shape[2]), out=D_mybest[:, :, k, :])
        if k:
            D_mybest_maxindices = D_mybest.permute(0, 1, 3, 2)[
                torch.arange(D_mybest.shape[0], dtype=sets.dtype, device=sets.device).unsqueeze(1), 
                torch.arange(D_mybest.shape[1], dtype=sets.dtype, device=sets.device).unsqueeze(0), 
                sets[k],
                :k
            ]

            torch.rsqrt(1 - innerp(D_mybest_maxindices),
                        out=temp_F_k_k[:, :, 0])  # torch.exp(-1/2 * torch.log1p(-inp), temp_F_k_k[:, 0]) temp_F_k_k is gamma_k
            D_mybest_maxindices *= -temp_F_k_k  # minimal operations, exploit linearity D_mybest_maxindices becomes -gamma_k * c_(k-1)
            D_mybest[:, :, k, :] *= temp_F_k_k # gamma_k * g_(lambda_k)
            D_mybest[:, :, k, :, None].view(-1, XTX.shape[1], 1).baddbmm_(D_mybest[:, :, :k, :].permute(0, 1, 3, 2).view(-1, XTX.shape[1], k), D_mybest_maxindices[:, :, :, None].view(-1, k, 1)) # B is updated up to k th column

        temp_a_F = temp_F_k_k * torch.gather(projections, 2, sets[k, :, :, None]) # a_k in the paper (b, 1) -> (B, b, 1)
        
        normr2.sub_((temp_a_F * temp_a_F).squeeze(-1))

        projections -= temp_a_F * D_mybest[:, :, k, :] # update projection p^(k) = p^(k-1) - b_(:k) * a_k
        a_F[k] = temp_a_F
        if k:  # Could maybe get a speedup from triangular mat mul kernel.
            torch.bmm(D_mybest_maxindices[:, :, None, :].view(-1, 1, k), F[:, :, :k, :].view(-1, k, n_nonzero_coefs), out=F[:, :, k, None, :].view(-1, 1, n_nonzero_coefs)) 
            F[:, :, k, k] = temp_F_k_k[..., 0] # update F
    else: # FIXME: else branch will not execute if n_nonzero_coefs=0, so solutions is undefined.
        # Normal exit, used when tolerance=None.
        solutions = F.permute(0, 1, 3, 2) @ a_F.squeeze(-1).permute(1, 2, 0)[:, :, :, None]

    return sets.permute(1, 2, 0).to(torch.int32), solutions, normr2, normr2_init, None, None

def omp(X, y, n_nonzero_coefs=None, tol=None):
    XTX = torch.bmm(X.permute(0, 2, 1), X)
    sets, solutions, errors, kv_normr2, lengths, above_thres = omp_v0(X, y, XTX, n_nonzero_coefs, tol)

    sets = sets.squeeze(0)
    solutions = solutions.squeeze()
    if lengths is not None:
        lengths = lengths.squeeze(0)
    else:
        lengths = torch.full((y.shape[1],), n_nonzero_coefs)

    solutions = solutions.squeeze()
    
    data = torch.cat([solutions[i, :lengths[i]] for i in range(y.shape[1])]).to(torch.float8_e4m3fn)
    indices = torch.cat([sets[i, :lengths[i]] for i in range(y.shape[1])]).to(torch.int16)

    indptr = torch.zeros(y.shape[1] + 1, dtype=torch.int32, device=sets.device)
    indptr[1:] = lengths.cumsum(dim=0)

    return indptr, indices, data
