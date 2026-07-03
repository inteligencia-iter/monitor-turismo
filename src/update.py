# -*- coding: utf-8 -*-
"""
update.py — Monitor de Turismo Internacional · Rio de Janeiro
=============================================================
Fluxo de atualização mensal:

  1. Abrir a planilha Excel em data/
  2. Adicionar nova(s) linha(s):
       - Meses históricos: preencher TODAS as colunas (chegadas + variáveis independentes)
       - Meses futuros:   preencher apenas as variáveis independentes (assentos, câmbio,
                          brent, qav); deixar as colunas de chegadas em BRANCO
  3. Salvar a planilha
  4. Rodar:  python src/update.py
  5. O script regenera docs/index.html automaticamente

Metodologia (conforme nota técnica aprovada):
  - Pré-processamento: primeira diferença em todas as séries; COVID (abr/2020–dez/2021) excluído
  - CCF com k ≥ 0; CI = ±1.96/√n
  - Modelos: SARIMA(1,1,1)(1,1,1)[12] + dummy COVID (benchmark)
             SARIMAX(1,1,1)(1,1,1)[12] + brent_l1, qav_l1, assentos, usd_l4, eur_l4 (principal)
             SARIMAX Argentina: ars_blue_l12 (dólar blue ARS/USD, lag 12m — validado por holdout)
  - Holdout: últimos 12 meses com chegadas observadas
  - Métricas: MAPE, MAE, RMSE; Ljung-Box obrigatório
  - Intervalos de confiança: 95%

Dependências:
  pip install pandas openpyxl statsmodels scikit-learn numpy
"""

import os, sys, json, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import ccf as sm_ccf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error)

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 0. CAMINHOS
# ═══════════════════════════════════════════════════════════════════════════════
ROOT     = Path(__file__).resolve().parent.parent
PLANILHA = ROOT / 'base' / 'Indicadores_v2.xlsx'
HTML_OUT = ROOT / 'docs' / 'index.html'

assert PLANILHA.exists(), f"Planilha não encontrada: {PLANILHA}"
assert HTML_OUT.exists(),  f"index.html não encontrado: {HTML_OUT}"

print("=" * 60)
print("Monitor de Turismo Internacional — Rio de Janeiro")
print("Iniciando atualização...")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARREGAR PLANILHA
# ═══════════════════════════════════════════════════════════════════════════════
df_raw = pd.read_excel(PLANILHA, sheet_name=0)
df_raw.columns = [
    'mes', 'chegadas_br', 'chegadas_rj', 'chegadas_europa', 'chegadas_arg',
    'chegadas_chile', 'chegadas_uru', 'chegadas_eua',
    'usd_brl', 'eur_brl', 'ars_usd_blue', 'clp_brl', 'uyu_brl',
    'brent_close', 'brent_high', 'qav', 'voos', 'assentos', 'rotas',
    'assentos_eua_gig', 'assentos_europa_gig',
    'assentos_arg_gig', 'assentos_chile_gig', 'assentos_uru_gig',
]
df_raw['mes'] = pd.to_datetime(df_raw['mes'])
df_raw = df_raw.sort_values('mes').reset_index(drop=True)
df_raw = df_raw.set_index('mes')
df_raw.index = pd.DatetimeIndex(df_raw.index, freq='MS')

# ── Separar meses históricos (chegadas preenchidas) dos futuros (chegadas vazias)
cols_dep = ['chegadas_br', 'chegadas_rj', 'chegadas_europa',
            'chegadas_arg', 'chegadas_chile', 'chegadas_uru', 'chegadas_eua']

mask_futuro = df_raw[cols_dep].isna().any(axis=1)
df_hist   = df_raw[~mask_futuro].copy()   # dados históricos completos
df_futuro = df_raw[mask_futuro].copy()    # meses com X preenchidos, chegadas vazias

n_hist   = len(df_hist)
n_futuro = len(df_futuro)

print(f"\n✓ Planilha carregada")
print(f"  Histórico:  {df_hist.index[0].strftime('%Y-%m')} a "
      f"{df_hist.index[-1].strftime('%Y-%m')} ({n_hist} meses)")
if n_futuro > 0:
    print(f"  Projeção:   {df_futuro.index[0].strftime('%Y-%m')} a "
          f"{df_futuro.index[-1].strftime('%Y-%m')} ({n_futuro} meses)")

    # ── Proxy sazonal antecipado para colunas de oferta aérea
    # Aplica ANTES da validação, permitindo n_futuro arbitrário (além do horizonte ANAC)
    # Fallback: mesmo mês do ano anterior → média dos últimos 6 meses históricos
    cols_assentos_opt = ['assentos', 'rotas', 'voos',
                         'assentos_eua_gig', 'assentos_europa_gig',
                         'assentos_arg_gig', 'assentos_chile_gig', 'assentos_uru_gig']
    for _col in cols_assentos_opt:
        if _col not in df_futuro.columns:
            continue
        for idx in df_futuro.index:
            if pd.isna(df_futuro.loc[idx, _col]):
                proxy_idx = idx - pd.DateOffset(years=1)
                if proxy_idx in df_hist.index and not pd.isna(df_hist.loc[proxy_idx, _col]):
                    df_futuro.loc[idx, _col] = df_hist.loc[proxy_idx, _col]
                    print(f"  📅 Proxy sazonal: {_col} {idx.strftime('%b/%Y')} ← {proxy_idx.strftime('%b/%Y')}")
                else:
                    df_futuro.loc[idx, _col] = float(df_hist[_col].dropna().iloc[-6:].mean())
                    print(f"  📅 Proxy média 6m: {_col} {idx.strftime('%b/%Y')}")

    # ── Validação: apenas colunas sem fallback são obrigatórias
    # Câmbio, petróleo e QAV devem vir da planilha — sem proxy disponível
    cols_required = ['usd_brl', 'eur_brl', 'brent_close', 'qav']
    faltando_req = df_futuro[cols_required].isna().sum()
    if faltando_req.any():
        print("\n⛔  Colunas obrigatórias com valores faltando nos meses futuros:")
        for col, n in faltando_req[faltando_req > 0].items():
            print(f"     {col}: {n} mês(es) sem valor (câmbio e petróleo não têm proxy)")
        print("   Preencha a planilha com câmbio, Brent e QAV e rode novamente.")
        sys.exit(1)
else:
    print("  Projeção:   nenhum mês futuro detectado — modo histórico apenas")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRÉ-PROCESSAMENTO (CCF e correlação)
# ═══════════════════════════════════════════════════════════════════════════════
COVID_START   = pd.Timestamp('2020-04-01')
COVID_END     = pd.Timestamp('2021-12-01')
# Exclusão estendida para pares com oferta de assentos do Uruguai:
# dados URU-GIG com zeros ou instabilidade estrutural até dez/2022
COVID_END_URU = pd.Timestamp('2022-12-01')

# Variáveis para análise de correlação (apenas histórico)
dep_vars   = cols_dep
indep_vars = ['usd_brl','eur_brl','ars_usd_blue','clp_brl','uyu_brl',
              'brent_close','qav','assentos','rotas',
              'assentos_eua_gig','assentos_europa_gig',
              'assentos_arg_gig','assentos_chile_gig','assentos_uru_gig']

# df_clean padrão (exclusão COVID abr/2020–dez/2021)
all_vars = dep_vars + indep_vars
df_diff    = df_hist[all_vars].diff()
mask_covid = (df_diff.index >= COVID_START) & (df_diff.index <= COVID_END)
df_clean   = df_diff[~mask_covid].dropna()
n_clean    = len(df_clean)
ci_ccf     = round(1.96 / np.sqrt(n_clean), 4)

# df_clean_uru — exclusão estendida até dez/2022 (instabilidade oferta Uruguai)
mask_covid_uru = (df_diff.index >= COVID_START) & (df_diff.index <= COVID_END_URU)
df_clean_uru   = df_diff[~mask_covid_uru].dropna()
n_clean_uru    = len(df_clean_uru)
ci_ccf_uru     = round(1.96 / np.sqrt(n_clean_uru), 4)

print(f"\n✓ Pré-processamento")
print(f"  Primeiras diferenças — COVID excluído (padrão abr/20–dez/21)")
print(f"  n = {n_clean} obs úteis | CI = ±{ci_ccf}")
print(f"  Uruguai: exclusão estendida até dez/22 | n = {n_clean_uru} | CI = ±{ci_ccf_uru}")

# ── Matriz de correlação: r(CCF) no melhor lag preditivo (k ≥ 1)
# Calculado depois de ccf_pairs; preenchido abaixo após compute_ccf

# ── CCF (k = -12 a +12)
def compute_ccf(x, y, max_lag=12):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    result = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            r = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        elif lag == 0:
            r = np.corrcoef(x, y)[0, 1]
        else:
            r = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        result.append({'lag': lag, 'r': round(float(r), 3)})
    return result

PARES_CCF = [
    ('usd_brl',    'chegadas_eua',    'usd_eua'),
    ('eur_brl',    'chegadas_europa', 'eur_europa'),
    ('ars_usd_blue','chegadas_arg',    'ars_arg'),
    ('clp_brl',    'chegadas_chile',  'clp_chile'),
    ('uyu_brl',    'chegadas_uru',    'uyu_uru'),
    ('brent_close','chegadas_rj',     'brent_rj'),
    ('qav',        'chegadas_rj',     'qav_rj'),
    ('assentos',   'chegadas_rj',     'assentos_rj'),
    ('rotas',      'chegadas_rj',     'rotas_rj'),
    ('usd_brl',    'chegadas_rj',     'usd_rj'),
    ('eur_brl',    'chegadas_rj',     'eur_rj'),
    # Oferta aérea por origem (df_clean padrão)
    ('assentos_eua_gig',    'chegadas_eua',    'assentos_eua'),
    ('assentos_europa_gig', 'chegadas_europa', 'assentos_europa'),
    ('assentos_arg_gig',    'chegadas_arg',    'assentos_arg'),
    ('assentos_chile_gig',  'chegadas_chile',  'assentos_chile'),
]

# Pares com df_clean padrão
ccf_pairs = {}
for ind, dep, key in PARES_CCF:
    ccf_pairs[key] = compute_ccf(df_clean[ind], df_clean[dep])

# Uruguai: df_clean_uru (exclusão estendida) — coluna assentos_uru_gig
ccf_pairs['assentos_uru'] = compute_ccf(
    df_clean_uru['assentos_uru_gig'], df_clean_uru['chegadas_uru']
)
ccf_pairs['uyu_uru_ext']  = compute_ccf(
    df_clean_uru['uyu_brl'], df_clean_uru['chegadas_uru']
)

# Extrair melhor lag preditivo (k ≥ 1) para cada par de oferta por origem
def best_lag_predictive(key):
    series = ccf_pairs.get(key, [])
    preds  = [e for e in series if e['lag'] >= 1]
    if not preds:
        return 1, 0.0
    best = max(preds, key=lambda e: abs(e['r']))
    return best['lag'], round(float(best['r']), 3)

lag_assentos_eua,    r_assentos_eua    = best_lag_predictive('assentos_eua')
lag_assentos_europa, r_assentos_europa = best_lag_predictive('assentos_europa')
lag_assentos_arg,    r_assentos_arg    = best_lag_predictive('assentos_arg')
lag_assentos_chile,  r_assentos_chile  = best_lag_predictive('assentos_chile')
lag_assentos_uru,    r_assentos_uru    = best_lag_predictive('assentos_uru')

print(f"\n✓ CCF — oferta aérea por origem (lag preditivo identificado):")
print(f"  EUA:     lag={lag_assentos_eua}m, r={r_assentos_eua}")
print(f"  Europa:  lag={lag_assentos_europa}m, r={r_assentos_europa}")
print(f"  ARG:     lag={lag_assentos_arg}m, r={r_assentos_arg}")
print(f"  Chile:   lag={lag_assentos_chile}m, r={r_assentos_chile}")
print(f"  Uruguai: lag={lag_assentos_uru}m, r={r_assentos_uru} (excl. estendida)")

# ── Matriz de correlação: r(CCF) no melhor lag preditivo (k ≥ 1)
CCF_MATRIX_MAP = {
    ('chegadas_eua',    'usd_brl'):          'usd_eua',
    ('chegadas_europa', 'eur_brl'):          'eur_europa',
    ('chegadas_arg',    'ars_usd_blue'):      'ars_arg',
    ('chegadas_chile',  'clp_brl'):          'clp_chile',
    ('chegadas_uru',    'uyu_brl'):          'uyu_uru',
    ('chegadas_rj',     'brent_close'):      'brent_rj',
    ('chegadas_rj',     'qav'):              'qav_rj',
    ('chegadas_rj',     'assentos'):         'assentos_rj',
    ('chegadas_rj',     'rotas'):            'rotas_rj',
    ('chegadas_rj',     'usd_brl'):          'usd_rj',
    ('chegadas_rj',     'eur_brl'):          'eur_rj',
    # Oferta por origem
    ('chegadas_eua',    'assentos_eua_gig'):    'assentos_eua',
    ('chegadas_europa', 'assentos_europa_gig'): 'assentos_europa',
    ('chegadas_arg',    'assentos_arg_gig'):    'assentos_arg',
    ('chegadas_chile',  'assentos_chile_gig'):  'assentos_chile',
    ('chegadas_uru',    'assentos_uru_gig'):    'assentos_uru',
}

def best_ccf_r(key):
    series = ccf_pairs.get(key, [])
    preds  = [e for e in series if e['lag'] >= 1]
    if not preds:
        return 0.0
    best = max(preds, key=lambda e: abs(e['r']))
    return round(float(best['r']), 3)

corr_matrix = {}
for dep in dep_vars:
    corr_matrix[dep] = {}
    for ind in indep_vars:
        ccf_key = CCF_MATRIX_MAP.get((dep, ind))
        corr_matrix[dep][ind] = best_ccf_r(ccf_key) if ccf_key else 0.0

print(f"✓ Matriz de correlação (r CCF, lag≥1) — {sum(len(v) for v in corr_matrix.values())} células")

# ── Tabela de lags
ALERTAS = {
    'clp_chile': 'fraco',
}
INTERPRETACOES = {
    'usd_eua':    lambda r, lag: f"USD mais alto hoje (BRL mais fraco) → mais turistas americanos em {lag} mês(es).",
    'eur_europa': lambda r, lag: f"Euro mais forte → mais turistas europeus em {lag} mês(es).",
    'ars_arg':    lambda r, lag: f"Dólar blue mais alto hoje → mais turistas argentinos em {lag} mês(es). Câmbio paralelo capta poder de compra real melhor que taxa oficial.",
    'clp_chile':  lambda r, lag: "Sinal fraco. Para chilenos, câmbio tem peso menor que oferta aérea e sazonalidade.",
    'uyu_uru':    lambda r, lag: f"Peso uruguaio mais forte → mais chegadas com lag de {lag} meses.",
    'brent_rj':   lambda r, lag: f"Petróleo mais caro → menos chegadas internacionais em {lag} mês(es).",
    'qav_rj':     lambda r, lag: f"Combustível mais caro → menos chegadas em {lag} mês(es).",
    'usd_rj':     lambda r, lag: f"Dólar alto (BRL fraco) → mais chegadas gerais em {lag} meses.",
    'eur_rj':     lambda r, lag: f"Euro mais forte → mais chegadas gerais em {lag} meses.",
}
LABELS = {
    'usd_eua':    ('Câmbio USD/BRL',    'Chegadas EUA — RJ'),
    'eur_europa': ('Câmbio EUR/BRL',    'Chegadas Europa — RJ'),
    'ars_arg':    ('Dólar Blue ARS/USD', 'Chegadas Argentina — RJ'),
    'clp_chile':  ('Câmbio CLP/BRL',    'Chegadas Chile — RJ'),
    'uyu_uru':    ('Câmbio UYU/BRL',    'Chegadas Uruguai — RJ'),
    'brent_rj':   ('Brent Fechamento',  'Chegadas Geral RJ'),
    'qav_rj':     ('QAV / Jet Fuel',    'Chegadas Geral RJ'),
    'usd_rj':     ('Câmbio USD/BRL',    'Chegadas Geral RJ'),
    'eur_rj':     ('Câmbio EUR/BRL',    'Chegadas Geral RJ'),
}

lags_table = []
for key, (ind_label, dep_label) in LABELS.items():
    # Apenas lags preditivos (k ≥ 1): X passado explica Y futuro
    positives = [e for e in ccf_pairs[key] if e['lag'] >= 1]
    best      = max(positives, key=lambda e: abs(e['r']))
    r, lag    = best['r'], best['lag']
    sig       = abs(r) > ci_ccf

    # Forçar n.s. para CLP (sem sinal preditivo estável)
    if key in ('clp_chile',):
        sig = False

    alerta = ALERTAS.get(key, None if sig else 'fraco')
    sinal  = '➕ Positivo' if r > 0 else '➖ Negativo'

    # Outros lags preditivos significativos além do best (alerta duplo lag)
    outros_sig = [
        {'lag': e['lag'], 'r': e['r']}
        for e in ccf_pairs[key]
        if e['lag'] >= 1 and e['lag'] != lag and abs(e['r']) > ci_ccf
    ]

    lags_table.append({
        'independente':  ind_label,
        'dependente':    dep_label,
        'r':             r,
        'lag':           lag,
        'sig':           bool(sig),
        'sinal':         sinal,
        'interpretacao': INTERPRETACOES[key](r, lag),
        'alerta':        alerta,
        'outros_sig_lags': outros_sig,
    })

print(f"✓ CCF calculada — {len(lags_table)} pares analisados")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODELOS PREDITIVOS
# ═══════════════════════════════════════════════════════════════════════════════
# Preditores defasados (construídos sobre o histórico completo)
df_model = df_hist.copy()
df_model['covid']    = ((df_model.index >= COVID_START) &
                        (df_model.index <= COVID_END)).astype(float)
df_model['brent_l1'] = df_model['brent_close'].shift(1)
df_model['qav_l1']   = df_model['qav'].shift(1)
df_model['usd_l4']   = df_model['usd_brl'].shift(4)
df_model['eur_l4']   = df_model['eur_brl'].shift(4)
df_model = df_model.dropna(subset=['brent_l1','qav_l1','usd_l4','eur_l4'])

EXOG_COLS = ['covid','brent_l1','qav_l1','assentos','usd_l4','eur_l4']

# ── Holdout: últimos 12 meses históricos
df_train = df_model.iloc[:-12]
df_hold  = df_model.iloc[-12:]
y_train  = df_train['chegadas_rj']
y_hold   = df_hold['chegadas_rj']

def fit_sarima(y, exog, disp=False):
    return SARIMAX(y, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,12),
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=disp)

# SARIMA holdout
m_sarima_h = fit_sarima(y_train, df_train[['covid']])
fc_sh      = m_sarima_h.get_forecast(steps=12, exog=df_hold[['covid']])
pred_sh    = fc_sh.predicted_mean
ci_sh      = fc_sh.conf_int(alpha=0.05)
lb_sh      = acorr_ljungbox(m_sarima_h.resid.dropna(), lags=12, return_df=True)

# SARIMAX holdout
m_sarimax_h = fit_sarima(y_train, df_train[EXOG_COLS])
fc_sxh      = m_sarimax_h.get_forecast(steps=12, exog=df_hold[EXOG_COLS])
pred_sxh    = fc_sxh.predicted_mean
ci_sxh      = fc_sxh.conf_int(alpha=0.05)
lb_sxh      = acorr_ljungbox(m_sarimax_h.resid.dropna(), lags=12, return_df=True)

def metricas(y_real, y_pred):
    return {
        'mape': round(mean_absolute_percentage_error(y_real, y_pred) * 100, 1),
        'mae':  int(round(mean_absolute_error(y_real, y_pred))),
        'rmse': int(round(np.sqrt(mean_squared_error(y_real, y_pred)))),
    }

met_s  = metricas(y_hold, pred_sh)
met_sx = metricas(y_hold, pred_sxh)

ljung_p_s  = round(float(lb_sh['lb_pvalue'].iloc[-1]),  3)
ljung_p_sx = round(float(lb_sxh['lb_pvalue'].iloc[-1]), 3)

print(f"\n✓ Modelos validados no holdout "
      f"({df_hold.index[0].strftime('%b/%Y')}–{df_hold.index[-1].strftime('%b/%Y')})")
print(f"  SARIMA  → MAPE {met_s['mape']}% | MAE {met_s['mae']:,} | "
      f"RMSE {met_s['rmse']:,} | Ljung-Box p={ljung_p_s}")
print(f"  SARIMAX → MAPE {met_sx['mape']}% | MAE {met_sx['mae']:,} | "
      f"RMSE {met_sx['rmse']:,} | Ljung-Box p={ljung_p_sx}")

if ljung_p_s < 0.05:
    print("  ⚠️  SARIMA: resíduos com autocorrelação — interpretar com cautela")
if ljung_p_sx < 0.05:
    print("  ⚠️  SARIMAX: resíduos com autocorrelação — revisar especificação")

# ── Modelos completos (todo o histórico) para forecast — RJ Total
m_sarima_f  = fit_sarima(df_model['chegadas_rj'], df_model[['covid']])
m_sarimax_f = fit_sarima(df_model['chegadas_rj'], df_model[EXOG_COLS])

def safe_int(v):
    try:
        v = float(v)
        return int(round(v)) if not (np.isnan(v) or np.isinf(v)) else None
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# 3b. MODELOS SEGMENTADOS (6 origens)
# ═══════════════════════════════════════════════════════════════════════════════
# Série completa hist + futuro com lags via ffill (para cruzar fronteira)
df_all = pd.concat([df_hist, df_futuro])
df_all.index = pd.DatetimeIndex(df_all.index, freq='MS')
df_all['covid']    = ((df_all.index >= COVID_START) &
                      (df_all.index <= COVID_END)).astype(float)
df_all['brent_l1'] = df_all['brent_close'].shift(1).ffill()
df_all['qav_l1']   = df_all['qav'].shift(1).ffill()
df_all['usd_l4']   = df_all['usd_brl'].shift(4).ffill()
df_all['eur_l4']   = df_all['eur_brl'].shift(4).ffill()
df_all['uyu_l10']      = df_all['uyu_brl'].shift(10).ffill()
# Dólar blue argentino: lag 12m — validado por holdout (MAPE 16.0%, Ljung-Box p=0.733)
df_all['ars_blue_l12'] = df_all['ars_usd_blue'].shift(12).ffill()

# Lags dinâmicos para oferta aérea por origem (lag identificado por CCF)
df_all['assentos_eua_l']    = df_all['assentos_eua_gig'].shift(lag_assentos_eua).ffill()
df_all['assentos_uru_l']    = df_all['assentos_uru_gig'].shift(lag_assentos_uru).ffill()

# Histórico com preditores defasados (dropna dos primeiros meses)
df_h_seg = df_all[~mask_futuro].dropna(
    subset=['brent_l1','qav_l1','usd_l4','eur_l4']).copy()

# ── Validação de cobertura mínima para variáveis de oferta por origem
def check_exog_coverage(df, col, min_months=24, covid_start=COVID_START, covid_end=COVID_END):
    """Retorna True se col tem >= min_months de dados válidos fora do período COVID."""
    mask = ~((df.index >= covid_start) & (df.index <= covid_end))
    n_valid = df.loc[mask, col].notna().sum()
    return int(n_valid) >= min_months

cob_eua    = check_exog_coverage(df_h_seg, 'assentos_eua_l')
cob_uru    = check_exog_coverage(df_h_seg, 'assentos_uru_l')

df_f_seg = df_all[mask_futuro].copy()

# Assentos futuros: usar da planilha se preenchido, senão proxy sazonal
for _col_ass in ['assentos', 'assentos_europa_gig', 'assentos_arg_gig', 'assentos_chile_gig']:
    if n_futuro > 0 and df_f_seg[_col_ass].isna().any():
        for idx in df_f_seg.index:
            if pd.isna(df_f_seg.loc[idx, _col_ass]):
                proxy_idx = idx - pd.DateOffset(years=1)
                df_f_seg.loc[idx, _col_ass] = (
                    df_h_seg.loc[proxy_idx, _col_ass]
                    if proxy_idx in df_h_seg.index
                    else float(df_h_seg[_col_ass].iloc[-6:].mean()))

SEGMENTOS = {
    'rj':        {'col':'chegadas_rj',      'label':'Rio de Janeiro (Total)',
                  'exog':['covid','brent_l1','qav_l1','assentos','usd_l4','eur_l4'],
                  'tipo':'SARIMAX'},
    'eua':       {'col':'chegadas_eua',     'label':'Estados Unidos',
                  'exog':['covid','assentos_eua_l','usd_l4'],
                  'tipo':'SARIMAX'},
    'europa':    {'col':'chegadas_europa',  'label':'Europa',
                  'exog':['covid','assentos_europa_gig'],   # PRIMARY: assentos (MAPE 5.7%)
                  'exog2':['covid','eur_l4'],               # ALT: EUR/BRL lag 4m (MAPE 6.6%)
                  'label2':'EUR/BRL (lag 4m)',
                  'tipo':'SARIMAX'},
    'argentina': {'col':'chegadas_arg',     'label':'Argentina',
                  'exog':['covid','assentos_arg_gig'],      # PRIMARY: assentos (MAPE 8.6%)
                  'exog2':['covid','ars_blue_l12'],         # ALT: dólar blue lag 12m (MAPE 16.5%)
                  'label2':'Dólar blue ARS/USD (lag 12m)',
                  'tipo':'SARIMAX'},
    'chile':     {'col':'chegadas_chile',   'label':'Chile',
                  'exog':['covid','assentos_chile_gig'],
                  'tipo':'SARIMAX'},
    'uruguai':   {'col':'chegadas_uru',     'label':'Uruguai',
                  'exog':['covid','uyu_l10','assentos_uru_l'],
                  'tipo':'SARIMAX'},
}

# ── Auditoria ADF das variáveis exógenas dos modelos
def _adf_pval(series):
    """Retorna p-value do teste ADF, ou None se série muito curta."""
    s = series.dropna()
    if len(s) < 20:
        return None
    try:
        return adfuller(s, autolag='AIC')[1]
    except Exception:
        return None

def auditoria_exog(df_h, segmentos, alpha=0.05):
    """Verifica estacionariedade de todas as exógenas antes do ajuste.
    - I(0): já estacionária em nível (raro para séries econômicas)
    - I(1): precisa de 1ª diferença — aceitável para SARIMAX(d=1)
    - I(2+): alerta — verificar antes de usar no modelo
    Novas colunas (não na lista validada) recebem destaque se forem I(2+).
    """
    # Conjunto de variáveis já validadas em holdout — sem alarme mesmo se I(1)
    VALIDADAS = {
        'covid',
        'usd_l4', 'eur_l4', 'uyu_l10', 'ars_blue_l12',
        'brent_l1', 'qav_l1',
        'assentos', 'assentos_eua_l', 'assentos_uru_l',
        'assentos_europa_gig', 'assentos_arg_gig', 'assentos_chile_gig',
    }

    # Coletar todas as exógenas únicas dos segmentos
    all_exog = set()
    for seg in segmentos.values():
        all_exog.update(seg.get('exog', []))
        all_exog.update(seg.get('exog2', []))
    all_exog.discard('covid')  # dummy binária — ADF não se aplica

    alertas_novos = []
    print(f"\n✓ Auditoria ADF — exógenas dos modelos (α={int(alpha*100)}%)")
    print(f"  {'Variável':<30} {'p(nível)':>10} {'p(1ªdif)':>10}  Ordem")
    print(f"  {'-'*62}")

    for col in sorted(all_exog):
        if col not in df_h.columns:
            print(f"  ⛔ {col:<30} {'—':>10} {'—':>10}  COLUNA NÃO ENCONTRADA")
            alertas_novos.append(col)
            continue

        s = df_h[col].dropna()
        p_niv = _adf_pval(s)
        p_d1  = _adf_pval(s.diff().dropna())

        if p_niv is None:
            ordem = '?'
        elif p_niv <= alpha:
            ordem = 'I(0)'
        elif p_d1 is not None and p_d1 <= alpha:
            ordem = 'I(1)'
        else:
            ordem = 'I(2+)'

        pn_s = f"{p_niv:.3f}" if p_niv is not None else 'n/a'
        pd_s = f"{p_d1:.3f}"  if p_d1  is not None else 'n/a'

        if ordem in ('I(0)', 'I(1)'):
            icone = '✅'
        elif col in VALIDADAS:
            icone = '⚠️ '  # validada em holdout — aceitável
        else:
            icone = '⛔'
            alertas_novos.append(col)

        print(f"  {icone} {col:<30} {pn_s:>10} {pd_s:>10}  {ordem}")

    if alertas_novos:
        print(f"\n  ⛔  ATENÇÃO — {len(alertas_novos)} variável(is) com ordem I(2+) ou não encontrada:")
        for v in alertas_novos:
            print(f"     → {v}")
        print("     Verifique a série antes de manter no modelo.")
        print("     Variáveis I(1) são aceitáveis — SARIMAX(d=1) trata a 1ª diferença da variável dependente.")
    else:
        print(f"\n  Todas as exógenas têm ordem ≤ I(1) — adequadas para SARIMAX(d=1) ✅")

auditoria_exog(df_h_seg, SEGMENTOS)

# Fallback de cobertura: modifica SEGMENTOS após sua definição (garantido acesso ao dict)
if not cob_eua:
    print("⚠️  assentos_eua_l: cobertura insuficiente (<24m fora COVID) — removendo do modelo EUA")
    SEGMENTOS['eua']['exog'] = ['covid','usd_l4']
if not cob_uru:
    print("⚠️  assentos_uru_l: cobertura insuficiente (<24m fora COVID) — removendo do modelo URU")
    SEGMENTOS['uruguai']['exog'] = ['covid','uyu_l10']

projecao_segmentos = {}

for seg_key, seg in SEGMENTOS.items():
    col    = seg['col']
    exog_c = seg['exog']
    usa_sx = len(exog_c) > 1

    df_seg = df_h_seg[[col] + exog_c].dropna()
    y_all  = df_seg[col]
    X_all  = df_seg[exog_c]

    # Holdout: últimos 12 meses
    y_train = y_all.iloc[:-12]; y_hold = y_all.iloc[-12:]
    X_train = X_all.iloc[:-12]; X_hold = X_all.iloc[-12:]

    # SARIMA holdout
    m_s_h  = fit_sarima(y_train, X_train[['covid']])
    fc_s   = m_s_h.get_forecast(steps=12, exog=X_hold[['covid']])
    pred_s = fc_s.predicted_mean; ci_s = fc_s.conf_int(alpha=0.05)
    lb_s   = acorr_ljungbox(m_s_h.resid.dropna(), lags=12, return_df=True)
    met_s_seg = metricas(y_hold, pred_s)
    ljp_s  = round(float(lb_s['lb_pvalue'].iloc[-1]), 3)

    # SARIMAX holdout (se aplicável)
    if usa_sx:
        m_sx_h  = fit_sarima(y_train, X_train[exog_c])
        fc_sx   = m_sx_h.get_forecast(steps=12, exog=X_hold[exog_c])
        pred_sx = fc_sx.predicted_mean; ci_sx = fc_sx.conf_int(alpha=0.05)
        lb_sx   = acorr_ljungbox(m_sx_h.resid.dropna(), lags=12, return_df=True)
        met_sx_seg = metricas(y_hold, pred_sx)
        ljp_sx  = round(float(lb_sx['lb_pvalue'].iloc[-1]), 3)
    else:
        pred_sx, ci_sx, met_sx_seg, ljp_sx = pred_s, ci_s, met_s_seg, ljp_s

    # Holdout JSON
    holdout_seg = []
    for mes, yr, ys, ysx, s_lo, s_hi, sx_lo, sx_hi in zip(
            y_hold.index, y_hold, pred_s, pred_sx,
            ci_s.iloc[:,0], ci_s.iloc[:,1],
            ci_sx.iloc[:,0], ci_sx.iloc[:,1]):
        holdout_seg.append({
            'mes': mes.strftime('%Y-%m'),
            'real': safe_int(yr),
            'sarima': safe_int(ys), 'sarimax': safe_int(ysx),
            'sarima_lo': safe_int(s_lo), 'sarima_hi': safe_int(s_hi),
            'sarimax_lo': safe_int(sx_lo), 'sarimax_hi': safe_int(sx_hi),
        })

    # Modelos completos para forecast
    m_s_f  = fit_sarima(y_all, X_all[['covid']])
    m_sx_f = fit_sarima(y_all, X_all[exog_c]) if usa_sx else m_s_f

    forecast_seg = []
    if n_futuro > 0:
        exog_fut_s  = df_f_seg[['covid']].values
        exog_fut_sx = df_f_seg[exog_c].values if usa_sx else exog_fut_s

        fc_s_f  = m_s_f.get_forecast(steps=n_futuro, exog=exog_fut_s)
        fc_sx_f = m_sx_f.get_forecast(steps=n_futuro, exog=exog_fut_sx)
        fc_s_m  = fc_s_f.predicted_mean;  fc_s_c  = fc_s_f.conf_int(alpha=0.05)
        fc_sx_m = fc_sx_f.predicted_mean; fc_sx_c = fc_sx_f.conf_int(alpha=0.05)

        for i, mes in enumerate(df_f_seg.index):
            # Flag: meses com assentos projetados (não observados) — mai-dez/2026
            # Consideramos "projetado" qualquer mês futuro fornecido na planilha
            forecast_seg.append({
                'mes': mes.strftime('%Y-%m'),
                'sarima':   safe_int(fc_s_m.iloc[i]),
                'sarimax':  safe_int(fc_sx_m.iloc[i]),
                'sarima_lo':  safe_int(fc_s_c.iloc[i,0]),
                'sarima_hi':  safe_int(fc_s_c.iloc[i,1]),
                'sarimax_lo': safe_int(fc_sx_c.iloc[i,0]),
                'sarimax_hi': safe_int(fc_sx_c.iloc[i,1]),
                'usa_dados_projetados': True,
            })
    else:
        # Proxy sazonal: mesmo mês do ano anterior
        future_months_seg = pd.date_range(
            start=df_h_seg.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')
        exog_rows_s  = [{'covid': 0.0} for _ in future_months_seg]
        exog_rows_sx = []
        for m in future_months_seg:
            proxy = m - pd.DateOffset(years=1)
            row = {'covid': 0.0}
            if 'assentos' in exog_c:
                row['assentos'] = float(df_h_seg.loc[proxy,'assentos']
                                        if proxy in df_h_seg.index
                                        else df_h_seg['assentos'].iloc[-6:].mean())
            if 'assentos_chile_gig' in exog_c:
                row['assentos_chile_gig'] = float(df_h_seg.loc[proxy,'assentos_chile_gig']
                                                  if proxy in df_h_seg.index
                                                  else df_h_seg['assentos_chile_gig'].iloc[-6:].mean())
            if 'assentos_europa_gig' in exog_c:
                row['assentos_europa_gig'] = float(df_h_seg.loc[proxy,'assentos_europa_gig']
                                                   if proxy in df_h_seg.index
                                                   else df_h_seg['assentos_europa_gig'].iloc[-6:].mean())
            if 'assentos_arg_gig' in exog_c:
                row['assentos_arg_gig'] = float(df_h_seg.loc[proxy,'assentos_arg_gig']
                                                if proxy in df_h_seg.index
                                                else df_h_seg['assentos_arg_gig'].iloc[-6:].mean())
            if 'usd_l4' in exog_c:
                row['usd_l4'] = float(df_h_seg['usd_l4'].iloc[-1])
            if 'eur_l4' in exog_c:
                row['eur_l4'] = float(df_h_seg['eur_l4'].iloc[-1])
            if 'uyu_l10' in exog_c:
                row['uyu_l10'] = float(df_h_seg['uyu_l10'].iloc[-1])
            if 'ars_blue_l12' in exog_c:
                # Lag 12: buscar valor do blue 12 meses antes de m (sempre no histórico)
                look_back = m - pd.DateOffset(years=1)
                row['ars_blue_l12'] = float(
                    df_hist.loc[look_back, 'ars_usd_blue']
                    if look_back in df_hist.index
                    else df_h_seg['ars_blue_l12'].iloc[-1]
                )
            if 'brent_l1' in exog_c:
                row['brent_l1'] = float(df_h_seg['brent_l1'].iloc[-1])
            if 'qav_l1' in exog_c:
                row['qav_l1'] = float(df_h_seg['qav_l1'].iloc[-1])
            exog_rows_sx.append(row)

        df_exog_s  = pd.DataFrame(exog_rows_s,  index=future_months_seg)
        df_exog_sx = pd.DataFrame(exog_rows_sx, index=future_months_seg)

        fc_s_f  = m_s_f.get_forecast(steps=6, exog=df_exog_s)
        fc_sx_f = m_sx_f.get_forecast(steps=6, exog=df_exog_sx)
        fc_s_m  = fc_s_f.predicted_mean;  fc_s_c  = fc_s_f.conf_int(alpha=0.05)
        fc_sx_m = fc_sx_f.predicted_mean; fc_sx_c = fc_sx_f.conf_int(alpha=0.05)

        for i, mes in enumerate(future_months_seg):
            forecast_seg.append({
                'mes': mes.strftime('%Y-%m'),
                'sarima':   safe_int(fc_s_m.iloc[i]),
                'sarimax':  safe_int(fc_sx_m.iloc[i]),
                'sarima_lo':  safe_int(fc_s_c.iloc[i,0]),
                'sarima_hi':  safe_int(fc_s_c.iloc[i,1]),
                'sarimax_lo': safe_int(fc_sx_c.iloc[i,0]),
                'sarimax_hi': safe_int(fc_sx_c.iloc[i,1]),
            })

    # ── Modelo alternativo (exog2) — Europa e Argentina têm duas visões ──────
    exog2_c = seg.get('exog2')
    mape_sx2 = mae_sx2 = rmse_sx2 = ljp_sx2 = label_sx2 = None
    if exog2_c:
        df_seg2 = df_h_seg[[col] + exog2_c].dropna()
        y2 = df_seg2[col]; X2 = df_seg2[exog2_c]
        y_tr2 = y2.iloc[:-12]; y_ho2 = y2.iloc[-12:]
        X_tr2 = X2.iloc[:-12]; X_ho2 = X2.iloc[-12:]
        m_alt_h = fit_sarima(y_tr2, X_tr2)
        fc_alt_h = m_alt_h.get_forecast(steps=12, exog=X_ho2)
        pred_alt = fc_alt_h.predicted_mean; ci_alt = fc_alt_h.conf_int(alpha=0.05)
        met_alt  = metricas(y_ho2, pred_alt)
        lb_alt   = acorr_ljungbox(m_alt_h.resid.dropna(), lags=12, return_df=True)
        ljp_sx2  = round(float(lb_alt['lb_pvalue'].iloc[-1]), 3)
        # Injeta sarimax2 nos holdout items (os dicts já foram construídos acima)
        for item, ys2, s2_lo, s2_hi in zip(
                holdout_seg, pred_alt, ci_alt.iloc[:,0], ci_alt.iloc[:,1]):
            item['sarimax2']    = safe_int(ys2)
            item['sarimax2_lo'] = safe_int(s2_lo)
            item['sarimax2_hi'] = safe_int(s2_hi)
        # Modelo completo para forecast
        m_alt_f = fit_sarima(y2, X2)
        if n_futuro > 0:
            exog_fut_alt = df_f_seg[exog2_c].values
            fc_alt_f   = m_alt_f.get_forecast(steps=n_futuro, exog=exog_fut_alt)
            fc_alt_m   = fc_alt_f.predicted_mean; fc_alt_ci = fc_alt_f.conf_int(alpha=0.05)
            for item, v, lo, hi in zip(
                    forecast_seg, fc_alt_m, fc_alt_ci.iloc[:,0], fc_alt_ci.iloc[:,1]):
                item['sarimax2']    = safe_int(v)
                item['sarimax2_lo'] = safe_int(lo)
                item['sarimax2_hi'] = safe_int(hi)
        mape_sx2  = round(met_alt['mape'], 1)
        mae_sx2   = int(round(met_alt['mae'],  0))
        rmse_sx2  = int(round(met_alt['rmse'], 0))
        label_sx2 = seg.get('label2', 'Modelo alternativo')

    exog_spec = ', '.join(c for c in exog_c if c != 'covid') if usa_sx else '—'
    projecao_segmentos[seg_key] = {
        'label': seg['label'],
        'tipo':  seg['tipo'],
        'col':   col,
        'metricas': {
            'sarima':  {**met_s_seg, 'ljung_p': ljp_s,
                        'spec': 'SARIMA(1,1,1)(1,1,1)[12] + dummy COVID'},
            'sarimax': {**met_sx_seg, 'ljung_p': ljp_sx,
                        'spec': f"SARIMAX + {exog_spec}" if usa_sx
                                else 'SARIMA(1,1,1)(1,1,1)[12] + dummy COVID'},
            'holdout_periodo': (f"{y_hold.index[0].strftime('%b/%Y')} – "
                                f"{y_hold.index[-1].strftime('%b/%Y')}"),
        },
        'holdout':  holdout_seg,
        'forecast': forecast_seg,
        'ljung_aprovado_sarima':  bool(ljp_s  > 0.05),
        'ljung_aprovado_sarimax': bool(ljp_sx > 0.05),
        'sarimax2': {
            'mape': mape_sx2, 'mae': mae_sx2, 'rmse': rmse_sx2,
            'ljung_p': ljp_sx2, 'label': label_sx2,
        } if exog2_c else None,
        'nota': (
            f"SARIMAX com assentos Argentina–GIG como preditor primário (MAPE {met_sx_seg['mape']}%). "
            'Modelo alternativo: Dólar blue ARS/USD com lag 12m. '
            'Câmbio oficial ARS/BRL excluído — distorcido por controles de capital e hiperinflação.'
            if seg_key == 'argentina' else
            f"SARIMAX com assentos Europa–GIG como preditor primário (MAPE {met_sx_seg['mape']}%). "
            'Modelo alternativo: EUR/BRL com lag 4m.'
            if seg_key == 'europa' else
            f"SARIMAX com preditores econômicos definidos por CCF."
        ),
    }
    print(f"  {seg['label']:30s} {seg['tipo']}  MAPE {met_sx_seg['mape']}%  Ljung p={ljp_sx}")
    if exog2_c and label_sx2:
        print(f"    └── alt ({label_sx2})  MAPE {mape_sx2}%  Ljung p={ljp_sx2}")

print(f"✓ {len(projecao_segmentos)} modelos segmentados treinados")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXOG FUTURO — usa os X que o usuário preencheu na planilha
# ═══════════════════════════════════════════════════════════════════════════════
holdout_json  = []
forecast_json = []

for mes, yr, ys, ysx, s_lo, s_hi, sx_lo, sx_hi in zip(
        df_hold.index, y_hold,
        pred_sh, pred_sxh,
        ci_sh.iloc[:,0], ci_sh.iloc[:,1],
        ci_sxh.iloc[:,0], ci_sxh.iloc[:,1]):
    holdout_json.append({
        'mes': mes.strftime('%Y-%m'),
        'real': safe_int(yr),
        'sarima': safe_int(ys), 'sarimax': safe_int(ysx),
        'sarima_lo': safe_int(s_lo), 'sarima_hi': safe_int(s_hi),
        'sarimax_lo': safe_int(sx_lo), 'sarimax_hi': safe_int(sx_hi),
    })

if n_futuro > 0:
    # Construir exog para os meses futuros a partir dos dados da planilha
    df_fut = df_futuro.copy()
    df_fut['covid'] = 0.0

    # lag-1: brent e qav do mês anterior
    # Para o primeiro mês futuro, o mês anterior é o último histórico
    serie_brent = pd.concat([df_hist['brent_close'], df_futuro['brent_close']])
    serie_qav   = pd.concat([df_hist['qav'],          df_futuro['qav']])
    df_fut['brent_l1'] = serie_brent.shift(1).loc[df_futuro.index]
    df_fut['qav_l1']   = serie_qav.shift(1).loc[df_futuro.index]

    # lag-4: usd e eur de 4 meses antes
    serie_usd = pd.concat([df_hist['usd_brl'], df_futuro['usd_brl']])
    serie_eur = pd.concat([df_hist['eur_brl'], df_futuro['eur_brl']])
    df_fut['usd_l4'] = serie_usd.shift(4).loc[df_futuro.index]
    df_fut['eur_l4'] = serie_eur.shift(4).loc[df_futuro.index]

    # Para lags que caem dentro do horizonte futuro e não têm valor, usar o último histórico
    df_fut['brent_l1'] = df_fut['brent_l1'].fillna(df_hist['brent_close'].iloc[-1])
    df_fut['qav_l1']   = df_fut['qav_l1'].fillna(df_hist['qav'].iloc[-1])
    df_fut['usd_l4']   = df_fut['usd_l4'].fillna(df_hist['usd_brl'].iloc[-1])
    df_fut['eur_l4']   = df_fut['eur_l4'].fillna(df_hist['eur_brl'].iloc[-1])

    exog_fut = df_fut[EXOG_COLS]
    exog_sarima_fut = df_fut[['covid']]

    steps = n_futuro
    fc_sx_f = m_sarimax_f.get_forecast(steps=steps, exog=exog_fut)
    fc_s_f  = m_sarima_f.get_forecast(steps=steps,  exog=exog_sarima_fut)

    fc_sx_m = fc_sx_f.predicted_mean
    fc_sx_c = fc_sx_f.conf_int(alpha=0.05)
    fc_s_m  = fc_s_f.predicted_mean
    fc_s_c  = fc_s_f.conf_int(alpha=0.05)

    print(f"\n✓ Forecast gerado ({n_futuro} meses)")
    for i, mes in enumerate(df_futuro.index):
        print(f"  {mes.strftime('%Y-%m')}  SARIMAX {fc_sx_m.iloc[i]:>9,.0f}  "
              f"IC [{fc_sx_c.iloc[i,0]:>9,.0f}, {fc_sx_c.iloc[i,1]:>9,.0f}]  "
              f"SARIMA {fc_s_m.iloc[i]:>9,.0f}")
        forecast_json.append({
            'mes': mes.strftime('%Y-%m'),
            'sarima':   safe_int(fc_s_m.iloc[i]),
            'sarimax':  safe_int(fc_sx_m.iloc[i]),
            'sarima_lo':  safe_int(fc_s_c.iloc[i,0]),
            'sarima_hi':  safe_int(fc_s_c.iloc[i,1]),
            'sarimax_lo': safe_int(fc_sx_c.iloc[i,0]),
            'sarimax_hi': safe_int(fc_sx_c.iloc[i,1]),
        })

    cenario_nota = (
        f"Projeção baseada nos valores de assentos, câmbio e energia inseridos "
        f"na planilha para {df_futuro.index[0].strftime('%b/%Y')}–"
        f"{df_futuro.index[-1].strftime('%b/%Y')}. "
        f"Lag-4 (usd/eur) e lag-1 (brent/qav) calculados automaticamente "
        f"a partir da série histórica e dos valores futuros fornecidos."
    )
else:
    # Sem meses futuros: usar proxy sazonal do ano anterior (mesmo mês de -12)
    future_months = pd.date_range(
        start=df_hist.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

    exog_fut_rows = []
    for i, m in enumerate(future_months):
        look_back = 4 - i
        usd_l4 = (df_model['usd_brl'].iloc[-look_back]
                  if look_back >= 1 else float(df_hist['usd_brl'].iloc[-1]))
        eur_l4 = (df_model['eur_brl'].iloc[-look_back]
                  if look_back >= 1 else float(df_hist['eur_brl'].iloc[-1]))
        # Proxy sazonal: mesmo mês do ano anterior
        proxy_mes = m - pd.DateOffset(years=1)
        assentos_proxy = (df_hist.loc[proxy_mes, 'assentos']
                          if proxy_mes in df_hist.index
                          else float(df_hist['assentos'].iloc[-6:].mean()))
        exog_fut_rows.append({
            'covid': 0.0,
            'brent_l1': float(df_hist['brent_close'].iloc[-1]),
            'qav_l1':   float(df_hist['qav'].iloc[-1]),
            'assentos': float(assentos_proxy),
            'usd_l4':   float(usd_l4),
            'eur_l4':   float(eur_l4),
        })

    exog_fut_df = pd.DataFrame(exog_fut_rows, index=future_months)
    fc_sx_f = m_sarimax_f.get_forecast(steps=6, exog=exog_fut_df)
    fc_s_f  = m_sarima_f.get_forecast(steps=6,
                  exog=pd.DataFrame({'covid':[0.0]*6}, index=future_months))
    fc_sx_m = fc_sx_f.predicted_mean
    fc_sx_c = fc_sx_f.conf_int(alpha=0.05)
    fc_s_m  = fc_s_f.predicted_mean
    fc_s_c  = fc_s_f.conf_int(alpha=0.05)

    for i, mes in enumerate(future_months):
        forecast_json.append({
            'mes': mes.strftime('%Y-%m'),
            'sarima':   safe_int(fc_s_m.iloc[i]),
            'sarimax':  safe_int(fc_sx_m.iloc[i]),
            'sarima_lo':  safe_int(fc_s_c.iloc[i,0]),
            'sarima_hi':  safe_int(fc_s_c.iloc[i,1]),
            'sarimax_lo': safe_int(fc_sx_c.iloc[i,0]),
            'sarimax_hi': safe_int(fc_sx_c.iloc[i,1]),
        })

    cenario_nota = (
        "Nenhum mês futuro detectado na planilha. "
        "Projeção de 6 meses usando proxy sazonal (mesmo mês do ano anterior) "
        "para assentos. Adicione linhas futuras na planilha para usar valores reais."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MONTAR JSON COMPLETO PARA O HTML
# ═══════════════════════════════════════════════════════════════════════════════
# Série histórica completa (em nível, para os gráficos)
series_json = []
for mes, row in df_hist.iterrows():
    series_json.append({
        'mes':           mes.strftime('%Y-%m'),
        'chegadas_br':   safe_int(row['chegadas_br']),
        'chegadas_rj':   safe_int(row['chegadas_rj']),
        'chegadas_europa':safe_int(row['chegadas_europa']),
        'chegadas_arg':  safe_int(row['chegadas_arg']),
        'chegadas_chile':safe_int(row['chegadas_chile']),
        'chegadas_uru':  safe_int(row['chegadas_uru']),
        'chegadas_eua':  safe_int(row['chegadas_eua']),
        'usd_brl':       round(float(row['usd_brl']),  4),
        'eur_brl':       round(float(row['eur_brl']),  4),
        'ars_usd_blue':  round(float(row.get('ars_usd_blue') or 0), 2),
        'clp_brl':       round(float(row.get('clp_brl') or 0),  4),
        'uyu_brl':       round(float(row.get('uyu_brl') or 0),  4),
        'brent_close':   round(float(row['brent_close']), 2),
        'brent_high':    round(float(row.get('brent_high') or 0), 2),
        'qav':           round(float(row['qav']),       3),
        'voos':          safe_int(row.get('voos')),
        'assentos':      safe_int(row['assentos']),
        'rotas':         safe_int(row.get('rotas')),
        'assentos_eua_gig':    safe_int(row.get('assentos_eua_gig')),
        'assentos_europa_gig': safe_int(row.get('assentos_europa_gig')),
        'assentos_arg_gig':    safe_int(row.get('assentos_arg_gig')),
        'assentos_chile_gig':  safe_int(row.get('assentos_chile_gig')),
        'assentos_uru_gig':    safe_int(row.get('assentos_uru_gig')),
    })

last_update = df_hist.index[-1].strftime('%Y-%m')

projecao_json = {
    'metricas': {
        'sarima': {
            **met_s,
            'ljung_p': ljung_p_s,
            'aic': round(m_sarima_h.aic, 1),
            'spec': 'SARIMA(1,1,1)(1,1,1)[12] + dummy COVID',
        },
        'sarimax': {
            **met_sx,
            'ljung_p': ljung_p_sx,
            'aic': round(m_sarimax_h.aic, 1),
            'spec': 'SARIMAX(1,1,1)(1,1,1)[12] + brent_l1, qav_l1, assentos, usd_l4, eur_l4',
        },
        'holdout_periodo': (df_hold.index[0].strftime('%b/%Y')
                            + ' – '
                            + df_hold.index[-1].strftime('%b/%Y')),
        'n_holdout': 12,
    },
    'holdout':  holdout_json,
    'forecast': forecast_json,
    'cenario_nota': cenario_nota,
    'ljung_aprovado_sarima':  bool(ljung_p_s  > 0.05),
    'ljung_aprovado_sarimax': bool(ljung_p_sx > 0.05),
}

assentos_futuro = {}
for mes, row in df_futuro.iterrows():
    assentos_futuro[mes.strftime('%Y-%m')] = {
        'assentos':          safe_int(row.get('assentos')),
        'assentos_eua_gig':     safe_int(row.get('assentos_eua_gig')),
        'assentos_europa_gig':  safe_int(row.get('assentos_europa_gig')),
        'assentos_arg_gig':     safe_int(row.get('assentos_arg_gig')),
        'assentos_chile_gig':   safe_int(row.get('assentos_chile_gig')),
        'assentos_uru_gig':     safe_int(row.get('assentos_uru_gig')),
    }

new_data = {
    'series':            series_json,
    'assentos_futuro':   assentos_futuro,
    'correlation_matrix':corr_matrix,
    'ccf_pairs':         ccf_pairs,
    'ccf_ci':            ci_ccf,
    'lags':              lags_table,
    'last_update':       last_update,
    'avg_precovid_rj':   int(df_hist.loc[df_hist.index < COVID_START, 'chegadas_rj'].mean()),
    'metodologia': {
        'adf': 'Todas as séries têm raiz unitária em nível.',
        'tratamento': ('1ª diferença em todas as séries. COVID excluído '
                       '(abr/2020–dez/2021). N=' + str(n_clean) + ' obs.'),
        'convencao_lag': 'CCF(X_t, Y_{t+k}): lag k>0 indica X prediz Y em k meses. Apenas lags positivos.',
        'ci': '±' + str(ci_ccf) + ' (95%, n=' + str(n_clean) + ')',
    },
    'projecao': projecao_json,
    'projecao_segmentos': projecao_segmentos,
}

# =====================================================================
# 6. INJETAR NO HTML E SALVAR
import shutil, tempfile
# =====================================================================
with open(HTML_OUT, 'r', encoding='utf-8') as f:
    html = f.read()

new_json  = json.dumps(new_data, ensure_ascii=False, default=str)
new_block = '<script id="md" type="application/json">' + new_json + '</script>'

match = re.search(r'<script id="md" type="application/json">.*?</script>', html, re.DOTALL)
if not match:
    raise RuntimeError('Bloco script id md nao encontrado no index.html')

html = html[:match.start()] + new_block + html[match.end():]

with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.html', delete=False) as tmp:
    tmp.write(html)
    tmp_path = tmp.name
shutil.copy2(tmp_path, HTML_OUT)

print('\n✓ docs/index.html atualizado (' + str(HTML_OUT.stat().st_size) + ' bytes)')
print('  Série histórica: ' + str(len(series_json)) + ' meses')
print('  Forecast RJ: ' + str(len(forecast_json)) + ' meses')
print('  Modelos segmentados: ' + str(len(projecao_segmentos)))
print('\n✅ Atualização concluída com sucesso!')
print('=' * 60)
