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
    logger.info("🔮 INICIANDO O SCANNER S-RANK: MOTOR DE SOBREVIVÊNCIA 🔮")

    # 1. Mapeando os caminhos
    caminho_atual = Path(__file__).resolve().parent
    caminho_dados = caminho_atual / "Data" / "Processed" / "obt_turnover_preparada.csv"

    # ATUALIZADO: Apontando para o modelo novo de Cox
    caminho_modelo = caminho_atual / "Models" / "cox_turnover_model.pkl"

    # 2. Invocando o Pipeline de Tempo
    logger.info("Carregando o Motor do Tempo (CoxPHFitter)...")
    modelo_cox = joblib.load(caminho_modelo)

    # 3. Puxando a Base
    logger.info("Puxando a base de dados preparada...")
    df = pd.read_csv(caminho_dados)

    # 4. O Filtro Supremo: Apenas quem está ATIVO na empresa hoje (0)
    df_ativos = df[df['target_pediu_demissao'] == 0].copy()
    logger.info(f"Rastreando o risco de {len(df_ativos)} colaboradores ativos...")

    # ------------------------------------------------------------------
    # 5. PREPARAÇÃO DO MOTOR (Pegando o que já veio da Fase 3)
    # ------------------------------------------------------------------
    # O arquivo 'preparada.csv' já tem 'is_com_dependentes', 'faixa_idade' e 'faixa_salarial'.
    # Só precisamos criar as duas flags que a gente forjava lá no train.py:

    df_ativos['is_perfil_Nao_Mapeado'] = np.where(
        df_ativos['perfil_comportamental'].str.contains('Não Mapeado', na=False), 1, 0
    )
    df_ativos['is_dep_RELACIONAMENTO'] = np.where(
        df_ativos['departamento_nome_api'] == 'RELACIONAMENTO', 1, 0
    )

    # Selecionando colunas base exatamente como o modelo aprendeu
    colunas_base = [
        'faixa_salarial', 'faixa_idade', 'is_com_dependentes',
        'is_perfil_Nao_Mapeado', 'is_dep_RELACIONAMENTO'
    ]
    X_ativos = df_ativos[colunas_base].copy()

    # Dummies (One-Hot Encoding)
    X_ativos = pd.get_dummies(X_ativos, columns=['faixa_salarial', 'faixa_idade'], drop_first=True)

    # 🛡️ JUTSU DE PROTEÇÃO: Alinhar as colunas para bater com o que o Cox aprendeu
    colunas_treino = modelo_cox.params_.index
    X_ativos = X_ativos.reindex(columns=colunas_treino, fill_value=0)

    # ------------------------------------------------------------------
    # 6. A MÁGICA DE PRODUÇÃO: O Radar de Degradação (Múltiplos Meses)
    # ------------------------------------------------------------------
    logger.info("Calculando o radar temporal (2, 3, 4 e 6 meses) para cada ativo...")

    curvas_sobrevivencia = modelo_cox.predict_survival_function(X_ativos)
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

            # Cálculo Condicional S-Rank
            chance_ficar = (p_futuro / p_atual) * 100 if p_atual > 0 else 0
            resultados_prob[h].append(round(chance_ficar, 1))

    # Injetando as colunas no DataFrame
    for h in horizontes:
        df_ativos[f'Chance_{h}M_%'] = resultados_prob[h]

    # ------------------------------------------------------------------
    # 7. TRADUÇÃO EXECUTIVA (Motor de Urgência)
    # ------------------------------------------------------------------
    # A IA lê a linha do tempo e diz o quão rápido o RH precisa correr.
    def classificar_urgencia(row):
        if row['Chance_2M_%'] <= 50: return '🚨 CRÍTICO (Ação HOJE)'
        if row['Chance_4M_%'] <= 50: return '🔴 ALERTA (Ação neste Trimestre)'
        if row['Chance_6M_%'] <= 50: return '🟡 ATENÇÃO (Monitorar Semestre)'
        if row['Chance_8M_%'] <= 50: return '🟡 ATENÇÃO (Monitorar 2° Semestre)'
        if row['Chance_10M_%'] <= 50: return '🟡 ATENÇÃO (Monitorar 2° Semestre)'
        if row['Chance_12M_%'] <= 50: return '🟡 ATENÇÃO (Monitorar ano completo)'

        return '🟢 SEGURO'

    df_ativos['Ação_RH'] = df_ativos.apply(classificar_urgencia, axis=1)

    # ------------------------------------------------------------------
    # 8. Montando a Fila de Prioridade (A Visão de Raio-X do RH)
    # ------------------------------------------------------------------
    df_relatorio = pd.DataFrame({
        'ID_Colaborador': df_ativos['colaborador_sk'],
        'Depto': df_ativos['departamento_nome_api'],
        'Meses_Casa': df_ativos['meses_de_casa'],
        'Ficar_2M': df_ativos['Chance_2M_%'].astype(str) + '%',
        'Ficar_3M': df_ativos['Chance_3M_%'].astype(str) + '%',
        'Ficar_4M': df_ativos['Chance_4M_%'].astype(str) + '%',
        'Ficar_6M': df_ativos['Chance_6M_%'].astype(str) + '%',
        'Ficar_8M': df_ativos['Chance_8M_%'].astype(str) + '%',
        'Ficar_10M': df_ativos['Chance_10M_%'].astype(str) + '%',
        'Ficar_12M': df_ativos['Chance_12M_%'].astype(str) + '%',
        'Urgência': df_ativos['Ação_RH']
    })

    # Ordenar pelos caras que tem a menor chance de sobreviver aos próximos 2 MESES
    # (O curto prazo é sempre a prioridade máxima na UTI do Turnover)
    df_relatorio = df_relatorio.iloc[df_ativos['Chance_2M_%'].argsort()][:15]

    print("\n" + "=" * 105)
    print("🚨 TARGET LIST S-RANK - MATRIZ DE DEGRADAÇÃO DE RETENÇÃO (TOP 15 URGÊNCIAS) 🚨")
    print("=" * 105)
    print(df_relatorio.to_string(index=False))

    # Salvando em Excel pra Reunião
    caminho_excel = caminho_atual / "Data" / "Processed" / "Target_List_RH_Evolutiva.xlsx"
    df_relatorio.to_excel(caminho_excel, index=False)
    logger.info(f"✅ Target List temporal gerada com sucesso em: {caminho_excel}")


if __name__ == "__main__":
    rodar_teste_real()