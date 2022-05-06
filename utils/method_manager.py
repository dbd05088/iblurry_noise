import logging

from methods.bic import BiasCorrection
from methods.er_baseline import ER
from methods.rainbow_memory import RM
from methods.ewc import EWCpp
from methods.mir import MIR
from methods.clib import CLIB
from methods.puridiver import PuriDivER
from methods.middle_loss import ML
from methods.clib_puridiver import CL_PURI
logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes, additional_trans=None):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "gdumb":
        from methods.gdumb import GDumb
        method = GDumb(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "rm":
        method = RM(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "bic":
        method = BiasCorrection(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "ewc++":
        method = EWCpp(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "mir":
        method = MIR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "clib":
        method = CLIB(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "ml":
        method = ML(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            additional_trans = additional_trans,
            **kwargs,
        )
    elif args.mode == "clib_puridiver":
        method = CL_PURI(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            additional_trans = additional_trans,
            **kwargs,
        )
    elif args.mode == "puridiver":
        method = PuriDivER(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            additional_trans = additional_trans,
            **kwargs,
        )
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method
