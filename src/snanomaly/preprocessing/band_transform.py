from __future__ import annotations

import numpy as np
import polars as pl
from attrs import define

from snanomaly.models.sncandidate import Bandset
from snanomaly.preprocessing.exc import ColumnNotFoundError


@define
class BandTransform:
    """
    Transform light curve data from one band set to another.

    Takes a Polars dataframe, finds the band set and the mean prediction values for the light curves, transforms
    the light curves from the source band set into the target band set, and returns the modified dataframe.

    Note: the `bandset` column is not updated to the converted band set to have a reference to the original one.
    """

    @classmethod
    def BRI_to_gri(cls, df: pl.DataFrame, bandset_col: str = "bandset", pred_means_col: str = "pred_means") -> pl.DataFrame:
        """
        The relevant Lupton's transformation equations are:
            B = g + 0.3130 (g - r) + 0.2271
            R = r - 0.1837 (g - r) - 0.0971
            R = r - 0.2936 (r - i) - 0.1439
            I = r - 1.2444 (r - i) - 0.3820
        """
        return cls._transform_dataframe_column(
            A=[[1.3130,-0.3130,0.],
               [-0.1837,1.1837,0.],
               [0.,0.7064,0.2936],
               [0.,-0.2444,1.2444]],
            b=[0.2271,-0.0971,-0.1439,-0.3820],
            df=df,
            bandset_col=bandset_col,
            pred_means_col=pred_means_col,
        )

    @classmethod
    def _transform_dataframe_column(
            cls,
            A: list[list[float]],
            b: list[float],
            df: pl.DataFrame,
            bandset_col: str = "bandset",
            pred_means_col: str = "pred_means",
    ) -> pl.DataFrame:
        """Applies the transformation to the dataframe specified by coefficient matrix `A` and constant vector `b`."""
        cols = df.columns
        for col in (bandset_col, pred_means_col):
            if col not in cols:
                raise ColumnNotFoundError(f"Column named `{col}` not found in dataframe")

        source_bandset = Bandset.BRI

        return df.with_columns(
            pl.col(pred_means_col)
            .where(pl.col(bandset_col) == source_bandset.value)
            .map_elements(
                lambda x: cls._transform(light_curves=x, A=A, b=b),
                return_dtype=list[list[float]],
            ),
        )

    @classmethod
    def _transform(cls, light_curves: list[list[float]], A: list[list[float]], b: list[float]) -> list[list[float]]:
        """
        Solves `Ax = b` for x.

        A: the coefficient matrix of the target bands (e.g.: 3 columns of A correspond to  g, r and i bands)
        x: the magnitudes of the target bands
        b: column vector for constants
        """
        num_equations = len(A)
        num_unknowns = len(light_curves)

        A = np.asarray(A)
        b = np.asarray(b)
        light_curves = np.asarray(light_curves)
        # convert fluxes to magnitudes
        light_curves = -2.5 * np.log10(light_curves)

        solve = cls._solve_least_square if num_equations > num_unknowns else cls._solve_linear
        transformed = solve(light_curves, A, b)
        # convert magnitudes to fluxes
        transformed = 10 ** (-0.4 * transformed)

        return transformed.tolist()

    @classmethod
    def _solve_linear(cls, light_curves: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(A, light_curves) + b

    @classmethod
    def _solve_least_square(cls, light_curves: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(A, light_curves - b, rcond=None)
