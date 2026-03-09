from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import sqlite3
import joblib
import subprocess
import sys
import os

app = FastAPI()

# --- CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS ---
# Pega a pasta onde o server.py está (Raiz do projeto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Montando o Front-End
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# 2. Carregando a Inteligência (Modelo e Dados)
def carregar_inteligencia():
    global modelo, df
    try:
        caminho_modelo = os.path.join(BASE_DIR, "Models", "lr_turnover_model.pkl")
        caminho_dados = os.path.join(BASE_DIR, "Data", "Processed", "obt_turnover_preparada.csv")

        modelo = joblib.load(caminho_modelo)
        df = pd.read_csv(caminho_dados)
        print("✅ Inteligência carregada com sucesso!")
    except Exception as e:
        print(f"⚠️ Erro ao carregar inteligência: {e}")


carregar_inteligencia()


# 3. Inicialização do Banco de Usuários (SQLite) - AGORA COM E-MAIL
def init_db():
    db_path = os.path.join(BASE_DIR, 'enterprise_users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Adicionamos a coluna email
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  email TEXT UNIQUE,
                  password TEXT, 
                  role TEXT)''')
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username, email, password, role) VALUES ('admin_rh', 'admin@arqdigital.com.br', '123456', 'Administrador')")
    conn.commit()
    conn.close()

init_db()

# 4. Schemas do Pydantic (Atualizados)
class UserLogin(BaseModel):
    username: str
    password: str

class NewUser(BaseModel):
    username: str
    email: str
    password: str
    role: str

# 5. Rotas de Autenticação e Gestão de Acessos (O Novo CRUD)
@app.post("/api/auth")
def login(creds: UserLogin):
    db_path = os.path.join(BASE_DIR, 'enterprise_users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE (username=? OR email=?) AND password=?", (creds.username, creds.username, creds.password))
    user = c.fetchone()
    conn.close()
    if user:
        return {"token": "enterprise_secure_token", "name": creds.username, "role": user[0]}
    raise HTTPException(status_code=401, detail="Credenciais Inválidas.")

@app.get("/api/users")
def get_users():
    db_path = os.path.join(BASE_DIR, 'enterprise_users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, username, email, role FROM users")
    users = [{"id": r[0], "username": r[1], "email": r[2], "role": r[3]} for r in c.fetchall()]
    conn.close()
    return users

@app.post("/api/users")
def create_user(user: NewUser):
    try:
        db_path = os.path.join(BASE_DIR, 'enterprise_users.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                  (user.username, user.email, user.password, user.role))
        conn.commit()
        conn.close()
        return {"msg": "Usuário criado com sucesso."}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Usuário ou E-mail já cadastrado.")

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int):
    db_path = os.path.join(BASE_DIR, 'enterprise_users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"msg": "Usuário removido."}

# 6. Rota S-Rank de Retreinamento de IA
@app.post("/api/retrain")
def retrain_model():
    try:
        caminho_main = os.path.join(BASE_DIR, "main.py")
        # Usando check=True e capturando a saída de erro detalhada
        result = subprocess.run(
            [sys.executable, caminho_main],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Se o main.py deu erro, a gente joga o erro real do Python no log do Render
            print(f"❌ ERRO NO MAIN.PY: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Erro interno no script: {result.stderr}")

        carregar_inteligencia()
        return {"msg": "Sucesso!"}
    except Exception as e:
        print(f"🔥 ERRO CRÍTICO NO SERVER: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 7. Rota de Inteligência de Dados (Adeus Dependentes, Olá Tempo Médio)
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

    # 🔥 A NOVA MATEMÁTICA: Média de Tempo de Casa (Ativos vs Evasões)
    media_tempo_ativos = round(ativos['meses_de_casa'].mean(), 1) if not ativos.empty else 0
    media_tempo_evasoes = round(demitidos['meses_de_casa'].mean(), 1) if not demitidos.empty else 0

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
        "tempo_medio": {"ativos": media_tempo_ativos, "evasoes": media_tempo_evasoes}, # O NOVO GRÁFICO
        "target_list": target_list.to_dict(orient="records")
    }