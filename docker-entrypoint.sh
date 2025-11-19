#!/bin/bash
set -e
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "==================================="
    echo "  LPR OCR System - Docker"
    echo "==================================="
    echo ""
    echo "Uso: docker-compose run lpr-ocr /data/placa.jpg"
    echo "Acurácia: 87.25%"
    exit 0
fi
[ $# -eq 0 ]: # "&& exec \"$0\" --help"
exec python -m lpr_ocr.predictor "$@"
