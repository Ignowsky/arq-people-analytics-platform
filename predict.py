import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report

# Importando o logger padrão do projeto
try:
    from Src.logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger("INFERENCIA_RH")


def recriar_features_faltantes(df_alvo):
    """
    Reconstrói as features matemáticas a partir do dado limpo.
    Assim não perdemos os nomes dos departamentos e cargos para o relatório final.
    """
    df_feat = df_alvo.copy()

    df_feat['is_perfil_Nao_Mapeado'] = np.where(df_feat['perfil_comportamental'].str.contains('Não Mapeado', na=False),
                                                1, 0)
    df_feat['is_dep_RELACIONAMENTO'] = np.where(df_feat['departamento_nome_api'] == 'RELACIONAMENTO', 1, 0)
    df_feat['is_com_dependentes'] = np.where(df_feat['qtd_dependentes'] > 0, 1, 0)

    df_feat['faixa_salarial'] = pd.qcut(df_feat['salario_contratual'], q=3, labels=['Baixo', 'Médio', 'Alto'])
    df_feat['faixa_idade'] = pd.cut(df_feat['idade'], bins=[0, 25, 35, 100], labels=['Ate_25', '26_a_35', 'Acima_35'])

    return df_feat


def auditoria_cenario_real(df, modelo_lr, modelo_cox):
    """
    Roda as métricas na base geral (histórica) para provar a acurácia
    da IA antes de prever o futuro dos ativos.
    """
    logger.info("🧪 INICIANDO AUDITORIA DE PRECISÃO (CENÁRIO REAL) 🧪")

    # 1. Aplica a engenharia de features em tempo real
    df_eval = recriar_features_faltantes(df)

    colunas_base = ['meses_de_casa', 'salario_contratual', 'idade', 'qtd_dependentes', 'faixa_salarial', 'faixa_idade',
                    'is_com_dependentes', 'is_perfil_Nao_Mapeado', 'is_dep_RELACIONAMENTO']
    colunas_presentes = [col for col in colunas_base if col in df_eval.columns]

    X_eval = df_eval[colunas_presentes].copy()
    X_eval = pd.get_dummies(X_eval, columns=['faixa_salarial', 'faixa_idade'], drop_first=True)

    # ---------------------------------------------------------
    # AVALIAÇÃO DA LOGÍSTICA
    # ---------------------------------------------------------
    X_lr_eval = X_eval.drop(columns=['meses_de_casa']) if 'meses_de_casa' in X_eval.columns else X_eval.copy()
    X_lr_eval = X_lr_eval.reindex(columns=modelo_lr.feature_names_in_, fill_value=0)

    y_true = df_eval['target_pediu_demissao']
    probabilidades = modelo_lr.predict_proba(X_lr_eval)[:, 1]

    # O Limiar de 39% que definimos no Treino
    y_pred = (probabilidades >= 0.39).astype(int)

    print('\n' + '=' * 70)
    print('--- PRECISÃO DO FRANCO-ATIRADOR (REGRESSÃO LOGÍSTICA - BASE GERAL) ---')
    print('=' * 70)
    print(classification_report(y_true, y_pred))

    # ---------------------------------------------------------
    # AVALIAÇÃO DO COX
    # ---------------------------------------------------------
    X_cox_eval = X_eval.reindex(columns=modelo_cox.params_.index, fill_value=0)

    df_cox_eval = X_cox_eval.copy()
    df_cox_eval['meses_de_casa'] = df_eval['meses_de_casa']
    df_cox_eval['target_pediu_demissao'] = df_eval['target_pediu_demissao']

    try:
        c_index = modelo_cox.score(df_cox_eval, scoring_method="concordance_index")
        print('=' * 70)
        print('--- ESTABILIDADE DO RADAR TEMPORAL (COX C-INDEX) ---')
        print('=' * 70)
        print(f'Concordance Index (Acerto do Relógio): {np.round(c_index, 2)}\n')
    except Exception as e:
        logger.warning(f"Não foi possível calcular o C-Index geral: {e}")


def rodar_teste_real():
    logger.info("🔮 INICIANDO O SCANNER S-RANK: ENSEMBLE (LOGÍSTICA + COX) 🔮")

    caminho_atual = Path(__file__).resolve().parent
    # 🚨 PONTO CHAVE: Apontando para o LIMPO obrigatoriamente
    caminho_dados = caminho_atual / "Data" / "Processed" / "obt_turnover_limpo.csv"
    caminho_modelo_cox = caminho_atual / "Models" / "cox_turnover_model.pkl"
    caminho_modelo_lr = caminho_atual / "Models" / "lr_turnover_model.pkl"

    logger.info("Carregando os Motores de IA...")
    modelo_cox = joblib.load(caminho_modelo_cox)
    modelo_lr = joblib.load(caminho_modelo_lr)

    logger.info("Puxando a base de dados preparada...")
    df = pd.read_csv(caminho_dados)

    # 🚀 CHAMADA DA AUDITORIA
    auditoria_cenario_real(df, modelo_lr, modelo_cox)

    # 4. O Filtro Supremo: Apenas quem está ATIVO na empresa hoje (0)
    df_ativos = df[df['target_pediu_demissao'] == 0].copy()
    logger.info(f"Rastreando o risco de {len(df_ativos)} colaboradores ativos...")

    # ------------------------------------------------------------------
    # 5. PREPARAÇÃO DA MATRIZ MATEMÁTICA (Usando a função ninja)
    # ------------------------------------------------------------------
    df_ativos = recriar_features_faltantes(df_ativos)

    colunas_base = ['meses_de_casa', 'salario_contratual', 'idade', 'qtd_dependentes', 'faixa_salarial', 'faixa_idade',
                    'is_com_dependentes', 'is_perfil_Nao_Mapeado', 'is_dep_RELACIONAMENTO']
    colunas_presentes = [col for col in colunas_base if col in df_ativos.columns]
    X_ativos = df_ativos[colunas_presentes].copy()
    X_ativos = pd.get_dummies(X_ativos, columns=['faixa_salarial', 'faixa_idade'], drop_first=True)

    colunas_treino_cox = modelo_cox.params_.index
    X_cox = X_ativos.reindex(columns=colunas_treino_cox, fill_value=0)

    X_lr = X_ativos.drop(columns=['meses_de_casa']) if 'meses_de_casa' in X_ativos.columns else X_ativos.copy()
    colunas_treino_lr = modelo_lr.feature_names_in_
    X_lr = X_lr.reindex(columns=colunas_treino_lr, fill_value=0)

    # ------------------------------------------------------------------
    # 6A. A MÁGICA 1: REGRESSÃO LOGÍSTICA
    # ------------------------------------------------------------------
    logger.info("Acionando a Regressão Logística (Risco Bruto)...")
    prob_fuga_lr = modelo_lr.predict_proba(X_lr)[:, 1] * 100

    df_lr_relatorio = pd.DataFrame({
        'ID_Colaborador': df_ativos['colaborador_sk'],
        'Depto': df_ativos['departamento_nome_api'],
        'Cargo': df_ativos.get('cargo_nome_api', 'Não Informado'),
        'Meses_Casa': df_ativos['meses_de_casa'],
        'Risco_Bruto_Fuga_%': np.round(prob_fuga_lr, 2)
    })
    df_lr_relatorio['Alerta_Imediato'] = np.where(df_lr_relatorio['Risco_Bruto_Fuga_%'] >= 39.0, '🚨 ALTO RISCO',
                                                  '🟢 SEGURO')
    df_lr_relatorio = df_lr_relatorio.sort_values(by='Risco_Bruto_Fuga_%', ascending=False)

    # ------------------------------------------------------------------
    # 6B. A MÁGICA 2: COX (Cálculo Contínuo Exato)
    # ------------------------------------------------------------------
    logger.info("Calculando o radar temporal (Cox) com interpolação exata da Lifelines...")

    horizontes = [2, 3, 4, 6, 8, 10, 12]
    resultados_prob = {h: [] for h in horizontes}

    tempos_necessarios = set()
    for idx in df_ativos.index:
        ta = df_ativos.loc[idx, 'meses_de_casa']
        tempos_necessarios.add(ta)
        for h in horizontes:
            tempos_necessarios.add(ta + h)

    tempos_ordenados = sorted(list(tempos_necessarios))

    curvas_sobrevivencia = modelo_cox.predict_survival_function(X_cox, times=tempos_ordenados)

    for idx in df_ativos.index:
        tempo_atual = df_ativos.loc[idx, 'meses_de_casa']
        p_atual = curvas_sobrevivencia.loc[tempo_atual, idx]

        for h in horizontes:
            tempo_futuro = tempo_atual + h
            p_futuro = curvas_sobrevivencia.loc[tempo_futuro, idx]

            chance_ficar = (p_futuro / p_atual) * 100 if p_atual > 0 else 0
            resultados_prob[h].append(round(chance_ficar, 1))

    # Injetando as colunas no DataFrame
    for h in horizontes:
        df_ativos[f'Chance_{h}M_%'] = resultados_prob[h]

    def classificar_urgencia(row):
        if row.get('Chance_2M_%', 100) <= 50: return '🚨 CRÍTICO (Ação HOJE)'
        if row.get('Chance_4M_%', 100) <= 50: return '🔴 ALERTA (Ação neste Trimestre)'
        if row.get('Chance_6M_%', 100) <= 50: return '🟡 ATENÇÃO (Monitorar Semestre)'
        if row.get('Chance_8M_%', 100) <= 50: return '🟡 ATENÇÃO (Monitorar 2° Semestre)'
        if row.get('Chance_10M_%', 100) <= 50: return '🟡 ATENÇÃO (Monitorar 2° Semestre)'
        if row.get('Chance_12M_%', 100) <= 50: return '🟡 ATENÇÃO (Monitorar ano completo)'
        return '🟢 SEGURO'

    df_ativos['Ação_RH'] = df_ativos.apply(classificar_urgencia, axis=1)

    # ------------------------------------------------------------------
    # 8. Montando a Fila de Prioridade
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

    df_relatorio = df_relatorio.iloc[df_ativos['Chance_2M_%'].argsort()][:15]

    print("\n" + "=" * 105)
    print("🚨 TARGET LIST S-RANK - MATRIZ DE DEGRADAÇÃO DE RETENÇÃO (TOP 15 URGÊNCIAS) 🚨")
    print("=" * 105)
    print(df_relatorio.to_string(index=False))

    caminho_excel = Path(
        "C:\\Users\\JoãoPedrodosSantosSa\\ARQDIGITAL LTDA\\RH DRIVE - Documentos\\9. ML & ANALYTICS\\Target List\\Target_List_ME.xlsx")
    os.makedirs(caminho_excel.parent, exist_ok=True)

    with pd.ExcelWriter(caminho_excel, engine='openpyxl') as writer:
        df_relatorio.to_excel(writer, sheet_name='Survival Analyses', index=False)
        df_lr_relatorio.to_excel(writer, sheet_name='Logistic Regression', index=False)

    logger.info(f"✅ Target List dupla gerada com sucesso em: {caminho_excel}")


if __name__ == "__main__":
    rodar_teste_real()