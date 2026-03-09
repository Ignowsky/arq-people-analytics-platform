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
    logger.info("🔮 INICIANDO O SCANNER S-RANK: PREVISÃO DE ATIVOS 🔮")

    # 1. Mapeando os caminhos
    caminho_atual = Path(__file__).resolve().parent
    caminho_dados = caminho_atual / "Data" / "Processed" / "obt_turnover_preparada.csv"
    caminho_modelo = caminho_atual / "Models" / "lr_turnover_model.pkl"

    # 2. Invocando o Pipeline Completo (Scaler + SMOTE + LR)
    logger.info("Carregando o Pipeline de Produção...")
    modelo_pipeline = joblib.load(caminho_modelo)

    # 3. Puxando a Base
    logger.info("Puxando a base de dados preparada...")
    df = pd.read_csv(caminho_dados)

    # 4. O Filtro Supremo: Apenas quem está ATIVO na empresa hoje (0)
    df_ativos = df[df['target_pediu_demissao'] == 0].copy()
    logger.info(f"Rastreando o risco de {len(df_ativos)} colaboradores ativos...")

    # 5. Engenharia de Atributos (Mimetizando o Treino)
    # IMPORTANTE: As colunas devem estar na MESMA ordem que o X_train foi alimentado
    df_ativos['is_perfil_Nao_Mapeado'] = np.where(
        df_ativos['perfil_comportamental'].str.contains('Não Mapeado', na=False), 1, 0
    )
    df_ativos['is_dep_RELACIONAMENTO'] = np.where(
        df_ativos['departamento_nome_api'] == 'RELACIONAMENTO', 1, 0
    )

    features_campeas = [
        'meses_de_casa',
        'salario_contratual',
        'idade',
        'qtd_dependentes',
        'is_perfil_Nao_Mapeado',
        'is_dep_RELACIONAMENTO'
    ]

    X_ativos = df_ativos[features_campeas]


    # 6. A Mágica de Produção: predict_proba
    # Como modelo_pipeline é um Pipeline do Sklearn, ele vai rodar o transform()
    # do StandardScaler automaticamente antes de prever!
    probabilidades = modelo_pipeline.predict_proba(X_ativos)[:, 1]


    # 7. Montando a Fila de Prioridade pro RH
    df_relatorio = pd.DataFrame({
        'ID_Colaborador': df_ativos['colaborador_sk'],
        'Departamento': df_ativos['departamento_nome_api'],
        'Perfil': df_ativos['perfil_comportamental'],
        'Tempo_Casa_Meses': df_ativos['meses_de_casa'],
        'Risco_Fuga (%)': (probabilidades * 100).round(2)
    })

    # Ordenar pelos que estão com o nível de estresse no talo
    df_relatorio = df_relatorio.sort_values(by='Risco_Fuga (%)', ascending=False).head(10)

    print("\n" + "=" * 70)
    print("🚨 TARGET LIST PRODUÇÃO - TOP 10 COLABORADORES ATIVOS EM RISCO 🚨")
    print("=" * 70)
    print(df_relatorio.head(10).to_string(index=False))

    # 8. Salvando em Excel pra levar pra reunião de diretoria
    caminho_excel = caminho_atual / "Data" / "Processed" / "Target_List_RH.xlsx"
    df_relatorio.to_excel(caminho_excel, index=False)
    logger.info(f"✅ Arquivo Excel gerado com sucesso para o RH em: {caminho_excel}")


if __name__ == "__main__":
    rodar_teste_real()