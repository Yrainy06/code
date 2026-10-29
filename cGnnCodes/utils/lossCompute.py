import torch


def MSE_loss(predicted, target):

    errors = predicted - target

    loss = (errors ** 2)

    return loss


def MAE_loss(predicted, target):

    errors = predicted - target

    loss = errors.abs()

    return loss


def huber_loss(predicted, target, delta=1.0):

    errors = predicted - target

    abs_err = errors.abs()
    quadratic = 0.5 * (abs_err ** 2)
    linear = delta * (abs_err - 0.5 * delta)
    loss = torch.where(abs_err < delta, quadratic, linear)

    return loss


def relative_huber_loss(predicted, target, eps=0.05, delta=1.0):

    errors = predicted - target

    denom  = target.abs() + eps
    r = errors / denom

    abs_r = r.abs()
    quad  = 0.5 * (abs_r ** 2)            # |r| < delta
    lin   = delta * (abs_r - 0.5 * delta) # |r| >= delta
    loss = torch.where(abs_r < delta, quad, lin)

    return loss


def Mach_re_loss(graph, predicted, target, squared=True):

    errors = predicted - target

    if squared:
        loss = (errors ** 2)
    else:
        loss = errors.abs()

    # Mach non-dimension
    # [rho, v1, v2, pres]
    mach = graph.x[:, 5]

    vm = mach.clone().detach()

    loss[:, 0] = loss[:, 0] / vm
    loss[:, 1] = loss[:, 1] / vm
    loss[:, 3] = loss[:, 3] / (vm * vm)

    return loss


def relative_loss(predicted, target, eps=0.05, squared=False):

    errors = predicted - target

    denom = target.abs() + eps

    if squared:
        loss = (errors ** 2) / (denom ** 2)
    else:
        loss = errors.abs() / denom

    return loss


def dataLoss(graph, predicted, target, mask):

    loss = MSE_loss(predicted, target)
    # loss = huber_loss(predicted, target)
    # loss = relative_huber_loss(predicted, target)
    # loss = relative_loss(predicted, target)
    # loss = Mach_re_loss(graph, predicted, target)

    loss = loss[mask]

    return loss.mean()
