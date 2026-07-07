"""Medição de impacto ambiental (energia e CO2) via CodeCarbon.

Terceira dimensão de métricas da metodologia. O rastreador é opcional: se o
CodeCarbon não estiver instalado (ou desabilitado na configuração), a pipeline
segue normalmente, apenas sem os indicadores energéticos.
"""

from __future__ import annotations

from contextlib import contextmanager

from config import Config


@contextmanager
def track_energy(cfg: Config, results: dict):
    """Context manager que mede a energia consumida no bloco envolvido.

    Uso:
        with track_energy(cfg, metrics):
            <computação a medir>
    Ao final, popula ``metrics`` com energia (kWh) e emissões (kg CO2).
    Como as medições de energia são ruidosas, recomenda-se envolver blocos de
    trabalho representativos (ex.: a avaliação completa sobre o conjunto de teste)
    e repetir o experimento com sementes distintas.
    """
    if not cfg.energy_enabled:
        yield
        return

    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        print("[energy] CodeCarbon indisponível — pulando medição de energia.")
        yield
        return

    tracker = EmissionsTracker(
        save_to_file=False,
        log_level="error",
        measure_power_secs=1,
    )
    tracker.start()
    try:
        yield
    finally:
        emissions_kg = tracker.stop()  # kg CO2eq
        results["energia_kwh"] = float(tracker.final_emissions_data.energy_consumed)
        results["emissoes_kg_co2"] = float(emissions_kg)
