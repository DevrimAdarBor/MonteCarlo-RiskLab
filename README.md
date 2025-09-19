# MonteCarlo-RiskLab — Monte Carlo Risk Lab

[![CI](https://github.com/DevrimAdarBor/MonteCarlo-RiskLab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DevrimAdarBor/MonteCarlo-RiskLab/actions)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Gerçek piyasa verileriyle kalibre edilen **Monte Carlo simülasyonu**.  

Desteklenen modeller:  
- **GBM** (Geometric Brownian Motion)  
- **t-GBM** (t-distributed GBM)  
- **Jump-Diffusion**  
- **GARCH** (uygunsa)  

Risk metrikleri:  
- **VaR** (Value-at-Risk)  
- **ES** (Expected Shortfall)  

Ek özellikler:  
- Portföy desteği  
- Backtest  
- Görselleştirme  

---

## Kurulum

```bash
pip install -r requirements.txt
## Örnek Kullanım
python mc.py --tickers AAPL MSFT --model tgbm --paths 20000 --years 1 --backtest --var-alpha 0.99
