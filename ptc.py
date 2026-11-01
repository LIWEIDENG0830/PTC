import torch


@torch.inference_mode()
def fit_projection_and_scores(dataset, device="cpu", batch_size=128, rtol=1e-7, ridge_scale=1e-6):
    dimension = dataset.seq_len + 1
    horizon = dataset.pred_len
    gram = torch.zeros(dimension, dimension, dtype=torch.float64)
    cross = torch.zeros(dimension, horizon, dtype=torch.float64)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        ones = torch.ones(x.shape[0], x.shape[2], 1, device=device)
        features = torch.cat((x.permute(0, 2, 1), ones), dim=2)
        targets = y.permute(0, 2, 1)
        gram += torch.einsum("bci,bcj->ij", features, features).double().cpu()
        cross += torch.einsum("bci,bch->ih", features, targets).double().cpu()
        rows += x.shape[0] * x.shape[2]
    theta = torch.linalg.pinv(gram / rows, rtol=rtol, hermitian=True) @ (cross / rows)
    ridge = max(1e-8, ridge_scale * float(torch.trace(gram)) / dimension)
    inverse = torch.linalg.inv(gram + ridge * torch.eye(dimension, dtype=torch.float64)).float().to(device)
    scores = []
    for x, _ in loader:
        x = x.to(device)
        ones = torch.ones(x.shape[0], x.shape[2], 1, device=device)
        features = torch.cat((x.permute(0, 2, 1), ones), dim=2)
        score = torch.einsum("bci,ij,bcj->b", features, inverse, features)
        scores.append(score.cpu())
    scores = torch.cat(scores).clamp_min(1e-12)
    if not torch.isfinite(scores).all() or scores.sum() <= 0:
        raise RuntimeError("Invalid leverage scores")
    return theta.float(), scores, ridge


def selection_orders(scores, maximum_k, random_seed=7, ptc_seed=17):
    if maximum_k < 1 or maximum_k > len(scores):
        raise ValueError(maximum_k)
    random_order = torch.randperm(len(scores), generator=torch.Generator().manual_seed(random_seed))[:maximum_k]
    probability = scores / scores.sum()
    uniform = torch.rand(len(scores), generator=torch.Generator().manual_seed(ptc_seed)).clamp_min(1e-12)
    ptc_order = torch.topk(torch.log(probability) - torch.log(-torch.log(uniform)), maximum_k).indices
    return {"random": random_order, "ptc": ptc_order}


def build_artifact(dataset, theta, indices, method):
    pairs = [dataset[int(index)] for index in indices]
    x = torch.stack([pair[0] for pair in pairs]).float()
    if method == "random":
        y = torch.stack([pair[1] for pair in pairs]).float()
    else:
        ones = torch.ones(x.shape[0], x.shape[2], 1)
        features = torch.cat((x.permute(0, 2, 1), ones), dim=2)
        y = (features @ theta).permute(0, 2, 1).float()
    return {"x": x, "y": y, "indices": indices.cpu(), "method": method, "k": len(indices)}
