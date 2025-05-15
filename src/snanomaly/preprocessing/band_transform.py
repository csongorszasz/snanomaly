from __future__ import annotations

import numpy as np
import polars as pl
from attrs import define

from snanomaly.models.sncandidate import Bandset
from snanomaly.preprocessing.exception import ColumnNotFoundError, InvalidTransformationTableError


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
            transform_table=[0, 1, 1, 2],
            df=df,
            bandset_col=bandset_col,
            pred_means_col=pred_means_col,
        )

    @classmethod
    def _transform_dataframe_column(
            cls,
            A: list[list[float]],
            b: list[float],
            transform_table: list[int],
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

        return df.with_columns([
            pl.when(pl.col(bandset_col) == source_bandset.value)
            .then(pl.col(pred_means_col).map_elements(
                lambda x: cls._transform(light_curves=x, A=A, b=b, transform_table=transform_table),
                return_dtype=list[list[float]],
            ))
            .otherwise(pl.col(pred_means_col))
            .alias(pred_means_col),
        ])

    @classmethod
    def _transform(
            cls,
            light_curves: list[list[float]] | pl.Series,
            A: list[list[float]],
            b: list[float],
            transform_table: list[int],
    ) -> list[list[float]]:
        """
        Solves `Ax = b` for x, where x is the transformed light curves.

        Parameters
        ----------
        A
            A list of lists that represents the coefficient matrix. Each column corresponds to a band.
        b
            A list of floats that represents the constant vector. Each element corresponds to a band.
        transform_table
            A list of integers that maps the columns of the coefficient matrix to the light curves.
            For example, if the coefficient matrix has 4 columns and the light curves have 3 bands = `BRI`,
            then `transform_table = [0, 1, 1, 2]` means that the first column of A corresponds to band `B`,
            the second column corresponds to `R`, the third column to `R` and the fourth to `I`.

        """
        num_equations = len(A)
        num_unknowns = len(light_curves)

        if min(transform_table) < 0 or max(transform_table) >= num_unknowns:
            raise InvalidTransformationTableError(f"Transform table values must be within [0, {num_unknowns}), but got {transform_table}")

        A = np.asarray(A)
        b = np.asarray(b)

        if isinstance(light_curves, pl.Series):
            light_curves = light_curves.to_list()
        light_curves = np.asarray(light_curves)
        light_curves = cls.fluxes_to_mags(light_curves)

        light_curves = light_curves[transform_table, :]

        solve = cls._solve_least_square if num_equations > num_unknowns else cls._solve_linear
        transformed = np.empty((num_unknowns, light_curves.shape[1]))
        for i in range(light_curves.shape[1]):
            transformed[:, i] = solve(light_curves[:, i], A, b)

        transformed = cls.mags_to_fluxes(transformed)
        transformed = transformed.tolist()

        return transformed

    @classmethod
    def fluxes_to_mags(cls, fluxes: np.ndarray) -> np.ndarray:
        mags = np.where(fluxes > 1e-8, fluxes, np.nan)
        return -2.5 * np.log10(mags)

    @classmethod
    def mags_to_fluxes(cls, mags: np.ndarray) -> np.ndarray:
        fluxes = 10 ** (-0.4 * mags)
        fluxes[np.isnan(fluxes)] = 0
        return fluxes

    @classmethod
    def _solve_linear(cls, light_curves: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(A, light_curves) + b

    @classmethod
    def _solve_least_square(cls, light_curves: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(A, light_curves - b, rcond=None)[0]
