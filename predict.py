import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Importando o logger padrão do projeto
try:
    from Src.logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger("INFERENCIA_RH")


def rodar_teste_real():
    logger.info("🔮 INICIANDO O SCANNER S-RANK: ENSEMBLE (LOGÍSTICA + COX) 🔮")

    # 1. Mapeando os caminhos
    caminho_atual = Path(__file__).resolve().parent
    caminho_dados = caminho_atual / "Data" / "Processed" / "obt_turnover_preparada.csv"

    # Apontando para os DOIS cérebros
    caminho_modelo_cox = caminho_atual / "Models" / "cox_turnover_model.pkl"
    caminho_modelo_lr = caminho_atual / "Models" / "lr_turnover_model.pkl"

    # 2. Invocando os Motores
    logger.info("Carregando os Motores de IA...")
    modelo_cox = joblib.load(caminho_modelo_cox)
    modelo_lr = joblib.load(caminho_modelo_lr)

    # 3. Puxando a Base
    logger.info("Puxando a base de dados preparada...")
    df = pd.read_csv(caminho_dados)

    # 4. O Filtro Supremo: Apenas quem está ATIVO na empresa hoje (0)
    df_ativos = df[df['target_pediu_demissao'] == 0].copy()
    logger.info(f"Rastreando o risco de {len(df_ativos)} colaboradores ativos...")

    # ------------------------------------------------------------------
    # 5. PREPARAÇÃO DA MATRIZ MATEMÁTICA
    # ------------------------------------------------------------------
    df_ativos['is_perfil_Nao_Mapeado'] = np.where(
        df_ativos['perfil_comportamental'].str.contains('Não Mapeado', na=False), 1, 0
    )
    df_ativos['is_dep_RELACIONAMENTO'] = np.where(
        df_ativos['departamento_nome_api'] == 'RELACIONAMENTO', 1, 0
    )

    # Incluídas as variáveis contínuas que os modelos precisam para não preencher com 0 no reindex
    colunas_base = [
        'meses_de_casa', 'salario_contratual', 'idade', 'qtd_dependentes',
        'faixa_salarial', 'faixa_idade', 'is_com_dependentes',
        'is_perfil_Nao_Mapeado', 'is_dep_RELACIONAMENTO'
    ]

    # Prevenção contra colunas ausentes
    colunas_presentes = [col for col in colunas_base if col in df_ativos.columns]
    X_ativos = df_ativos[colunas_presentes].copy()

    # Dummies (One-Hot Encoding)
    X_ativos = pd.get_dummies(X_ativos, columns=['faixa_salarial', 'faixa_idade'], drop_first=True)

    # 🛡️ JUTSU DE PROTEÇÃO (COX): Alinhar as colunas com o Treino
    colunas_treino_cox = modelo_cox.params_.index
    X_cox = X_ativos.reindex(columns=colunas_treino_cox, fill_value=0)

    # 🛡️ JUTSU DE PROTEÇÃO (LOGÍSTICA): Esconder o Relógio
    # O pipeline (SMOTE + LR) treinou sem a coluna 'meses_de_casa'
    X_lr = X_ativos.drop(columns=['meses_de_casa']) if 'meses_de_casa' in X_ativos.columns else X_ativos.copy()

    # ------------------------------------------------------------------
    # 6A. A MÁGICA 1: REGRESSÃO LOGÍSTICA (O Franco-Atirador)
    # ------------------------------------------------------------------
    logger.info("Acionando a Regressão Logística (Risco Bruto)...")

    # Extraindo a probabilidade (0 a 1) e convertendo para porcentagem
    prob_fuga_lr = modelo_lr.predict_proba(X_lr)[:, 1] * 100

    df_lr_relatorio = pd.DataFrame({
        'ID_Colaborador': df_ativos['colaborador_sk'],
        'Depto': df_ativos['departamento_nome_api'],
        'Cargo': df_ativos.get('cargo', 'Não Informado'),  # Pega o cargo se existir
        'Meses_Casa': df_ativos['meses_de_casa'],
        'Risco_Bruto_Fuga_%': np.round(prob_fuga_lr, 2)
    })

    # Aplicando o limiar de 39% que definimos no Treino
    df_lr_relatorio['Alerta_Imediato'] = np.where(df_lr_relatorio['Risco_Bruto_Fuga_%'] >= 39.0, '🚨 ALTO RISCO',
                                                  '🟢 SEGURO')
    df_lr_relatorio = df_lr_relatorio.sort_values(by='Risco_Bruto_Fuga_%', ascending=False)

    # ------------------------------------------------------------------
    # 6B. A MÁGICA 2: COX (O Radar de Degradação)
    # ------------------------------------------------------------------
    logger.info("Calculando o radar temporal (Cox) para múltiplos meses...")

    curvas_sobrevivencia = modelo_cox.predict_survival_function(X_cox)
    tempos_disponiveis = curvas_sobrevivencia.index.to_numpy()

    horizontes = [2, 3, 4, 6, 8, 10, 12]
    resultados_prob = {h: [] for h in horizontes}

    for idx in df_ativos.index:
        tempo_atual = df_ativos.loc[idx, 'meses_de_casa']
        t_atual = tempos_disponiveis[np.abs(tempos_disponiveis - tempo_atual).argmin()]
        p_atual = curvas_sobrevivencia.loc[t_atual, idx]

        for h in horizontes:
            tempo_futuro = tempo_atual + h
            t_futuro = tempos_disponiveis[np.abs(tempos_disponiveis - tempo_futuro).argmin()]
            p_futuro = curvas_sobrevivencia.loc[t_futuro, idx]

            chance_ficar = (p_futuro / p_atual) * 100 if p_atual > 0 else 0
            resultados_prob[h].append(round(chance_ficar, 1))

    for h in horizontes:
        df_ativos[f'Chance_{h}M_%'] = resultados_prob[h]

    def classificar_urgencia(row):
        if row.get('Chance_2M_%', 100) <= 50: return '🚨 CRÍTICO (Ação HOJE)'
        if row.get('Chance_4M_%', 100) <= 50: return '🔴 ALERTA (Ação neste Trimestre)'
        if row.get('Chance_6M_%', 100) <= 50: return '🟡 ATENÇÃO (Monitorar Semestre)'
        if row.get('Chance_8M_%', 100) <= 50: return '🟡 ATENÇÃO (Monitorar 2° Semestre)'
        if row.get('Chance_12M_%', 100) <= 50: return '🟡 ATENÇÃO (Monitorar ano completo)'
        return '🟢 SEGURO'

    df_ativos['Ação_RH'] = df_ativos.apply(classificar_urgencia, axis=1)

    df_cox_relatorio = pd.DataFrame({
        'ID_Colaborador': df_ativos['colaborador_sk'],
        'Depto': df_ativos['departamento_nome_api'],
        'Meses_Casa': df_ativos['meses_de_casa'],
        'Ficar_2M': df_ativos['Chance_2M_%'].astype(str) + '%',
        'Ficar_3M': df_ativos['Chance_3M_%'].astype(str) + '%',
        'Ficar_4M': df_ativos['Chance_4M_%'].astype(str) + '%',
        'Ficar_6M': df_ativos['Chance_6M_%'].astype(str) + '%',
        'Ficar_8M': df_ativos['Chance_8M_%'].astype(str) + '%',
        'Ficar_12M': df_ativos['Chance_12M_%'].astype(str) + '%',
        'Urgência': df_ativos['Ação_RH']
    })

    df_cox_relatorio = df_cox_relatorio.iloc[df_ativos['Chance_2M_%'].argsort()]

    # ------------------------------------------------------------------
    # 7. EXPORTAÇÃO S-RANK (Uma Planilha, Duas Abas)
    # ------------------------------------------------------------------
    caminho_excel = caminho_atual / "Data" / "Processed" / "Target_List_RH_Evolutiva.xlsx"
    os.makedirs(caminho_excel.parent, exist_ok=True)

    logger.info("Escrevendo os relatórios no Excel...")

    # O pd.ExcelWriter permite gravar várias planilhas no mesmo arquivo
    with pd.ExcelWriter(caminho_excel, engine='openpyxl') as writer:
        df_cox_relatorio.to_excel(writer, sheet_name='Survival Analyses', index=False)
        df_lr_relatorio.to_excel(writer, sheet_name='Logistic Regression', index=False)

    logger.info(f"✅ Target List dupla gerada com sucesso em: {caminho_excel}")


if __name__ == "__main__":
    rodar_teste_real()