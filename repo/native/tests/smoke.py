"""Teste rápido: carrega o módulo compilado e lista os drivers/dispositivos que o JUCE enxerga.
uso: python smoke.py PASTA_COM_O__native"""
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import _native as n  # noqa: E402

print(n.build_info())
print("tipos de driver:", n.AudioDevice.types())
for d in n.AudioDevice.list():
    print("  ", d.type, "|", d.name, "|", d.out_channels, "canais")
if "+JUCE" not in n.build_info():
    sys.exit("módulo compilado SEM JUCE")
