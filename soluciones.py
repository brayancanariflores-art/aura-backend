"""Presentación de un ejercicio para el chat web y para clientes de texto o voz."""
from pydantic import BaseModel, Field


class Paso(BaseModel):
    explicacion: str
    formula: str


class Desarrollo(BaseModel):
    pasos: list[Paso]
    exacto: str = Field(description="Resultado exacto en LaTeX con unidades. Vacío si no se conoce.")
    aproximacion: str = Field(description="Valor aproximado en LaTeX con ≈ y unidades, solo si aporta información.")
    nota: str = Field(description="Dato faltante, interpretación o salvedad necesaria. Vacío en otro caso.")
    lectura: str = Field(description="Respuesta legible en español y en texto plano para voz, con el resultado y sus unidades.")
