from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import sqlite3
from pathlib import Path
import joblib
import subprocess
import sys
import os

app = FastAPI()

# 1. Montando o Front-End
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Carregando a Inteligência (Modelo e Dados)
raiz = Path(__file__).resolve().parent


# Pega exatamente a pasta onde o server.py está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def carregar_inteligencia():
    global modelo, df
    # Junta os caminhos usando os.path.join para não dar pau no Linux/Windows
    caminho_modelo = os.path.join(BASE_DIR, "Models", "lr_turnover_model.pkl")
    caminho_dados = os.path.join(BASE_DIR, "Data", "Processed", "obt_turnover_preparada.csv")

    modelo = joblib.load(caminho_modelo)
    df = pd.read_csv(caminho_dados)

carregar_inteligencia()


# 3. Inicialização do Banco de Usuários (SQLite)
def init_db():
    conn = sqlite3.connect('enterprise_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     username
                     TEXT
                     UNIQUE,
                     password
                     TEXT,
                     role
                     TEXT
                 )''')
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username, password, role) VALUES ('admin_rh', '123456', 'Administrador')")
    conn.commit()
    conn.close()


init_db()


# 4. Schemas do Pydantic
class UserLogin(BaseModel):
    username: str
    password: str


class NewUser(BaseModel):
    username: str
    password: str
    role: str


# 5. Rotas de Autenticação e Gestão de Acessos
@app.post("/api/auth")
def login(creds: UserLogin):
    conn = sqlite3.connect('enterprise_users.db')
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (creds.username, creds.password))
    user = c.fetchone()
    conn.close()
    if user:
        return {"token": "enterprise_secure_token", "name": creds.username, "role": user[0]}
    raise HTTPException(status_code=401, detail="Credenciais Inválidas.")


@app.get("/api/users")
def get_users():
    conn = sqlite3.connect('enterprise_users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM users")
    users = [{"id": r[0], "username": r[1], "role": r[2]} for r in c.fetchall()]
    conn.close()
    return users


@app.post("/api/users")
def create_user(user: NewUser):
    try:
        conn = sqlite3.connect('enterprise_users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (user.username, user.password, user.role))
        conn.commit()
        conn.close()
        return {"msg": "Usuário criado com sucesso."}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Usuário já existe.")


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int):
    conn = sqlite3.connect('enterprise_users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"msg": "Usuário removido."}


# 6. Rota S-Rank de Retreinamento de IA
@app.post("/api/retrain")
def retrain_model():
    try:
        caminho_main = raiz / "main.py"
        subprocess.run([sys.executable, str(caminho_main)], check=True)
        carregar_inteligencia()  # Recarrega modelo e CSV novos na memória
        return {"msg": "Megazord retreinado com sucesso! A nova inteligência já está ativa."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro crítico no retreino: {str(e)}")


# 7. Rotas do Dashboard e Inteligência de Dados
@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.get("/api/departments")
def get_departments():
    deps = df['departamento_nome_api'].dropna().unique().tolist()
    return sorted(deps)


@app.get("/api/organizational_health")
def get_health_data(departamento: str = "Todos"):
    df_filtered = df.copy()
    if departamento != "Todos":
        df_filtered = df_filtered[df_filtered['departamento_nome_api'] == departamento]

    ativos = df_filtered[df_filtered['target_pediu_demissao'] == 0].copy()
    demitidos = df_filtered[df_filtered['target_pediu_demissao'] == 1].copy()

    hc_real = len(ativos)
    taxa_turnover = round((len(demitidos) / len(df_filtered) * 100), 1) if len(df_filtered) > 0 else 0

    hc_dep = ativos.groupby('departamento_nome_api').size()
    evasao_dep = demitidos.groupby('departamento_nome_api').size()
    idade_ativos = ativos['idade'].dropna().tolist()
    idade_demitidos = demitidos['idade'].dropna().tolist()

    col_genero = 'genero' if 'genero' in df_filtered.columns else 'perfil_comportamental'

    cols_num = ['meses_de_casa', 'salario_contratual', 'idade', 'qtd_dependentes', 'target_pediu_demissao']
    df_corr = df_filtered[cols_num].dropna()
    matriz_corr = df_corr.corr().round(2)
    nomes_eixos = ['Meses de Casa', 'Salário', 'Idade', 'Dependentes', 'Turnover']

    evasao_perfil = demitidos.groupby('perfil_comportamental').size()
    evasao_dep_familia = demitidos.groupby('qtd_dependentes').size()

    if not ativos.empty:
        ativos['is_perfil_Nao_Mapeado'] = ativos['perfil_comportamental'].apply(
            lambda x: 1 if 'Não Mapeado' in str(x) else 0)
        ativos['is_dep_RELACIONAMENTO'] = ativos['departamento_nome_api'].apply(
            lambda x: 1 if x == 'RELACIONAMENTO' else 0)
        cols = ['meses_de_casa', 'salario_contratual', 'idade', 'qtd_dependentes', 'is_perfil_Nao_Mapeado',
                'is_dep_RELACIONAMENTO']
        ativos['risco'] = (modelo.predict_proba(ativos[cols])[:, 1] * 100).round(2)
        target_list = ativos[['colaborador_sk', 'departamento_nome_api', 'perfil_comportamental', 'risco']].sort_values(
            'risco', ascending=False).head(50)
    else:
        target_list = pd.DataFrame()

    return {
        "kpis": {"headcount": hc_real, "evasoes": len(demitidos), "taxa_turnover": taxa_turnover},
        "departamentos": {"nomes": hc_dep.index.tolist(), "headcount": hc_dep.values.tolist(),
                          "evasoes": evasao_dep.reindex(hc_dep.index, fill_value=0).values.tolist()},
        "eda_avancada": {"genero": df_filtered[col_genero].tolist(),
                         "salario": df_filtered['salario_contratual'].tolist(),
                         "status": df_filtered['target_pediu_demissao'].tolist(),
                         "tempo": df_filtered['meses_de_casa'].tolist()},
        "demografia": {"idade_ativos": idade_ativos, "idade_demitidos": idade_demitidos},
        "correlacao": {"z": matriz_corr.values.tolist(), "eixos": nomes_eixos},
        "perfil": {"nomes": evasao_perfil.index.tolist(), "valores": evasao_perfil.values.tolist()},
        "dependentes": {"qtd": evasao_dep_familia.index.tolist(), "fugas": evasao_dep_familia.values.tolist()},
        "target_list": target_list.to_dict(orient="records")
    }