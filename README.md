# 🛡️ ARQ People Analytics - Turnover Prediction

> **"Prever o futuro não é mágica, é matemática aplicada com visão de negócio."**
> 
> Um ecossistema completo de *People Analytics* e *Machine Learning* construído para antecipar a evasão de talentos (Turnover) antes que a carta de demissão chegue ao RH.

Este projeto transcende um simples modelo de predição; trata-se de um **SaaS Corporativo Full-Stack**. Ele possui uma esteira MLOps autônoma (Backend) e uma Unidade de Inteligência de alta performance (Frontend) construída com design system premium (Dark Mode/Glassmorphism), protegida por autenticação e gestão de acessos via banco de dados relacional.

---

## 🚀 O Impacto de Negócio (Visão de Business Partner)

Perder um talento chave sangra o caixa da empresa (custos de rescisão, recrutamento, curva de aprendizado). O objetivo deste ecossistema é munir a diretoria (C-Level) e os BPs de RH com dados irrefutáveis e alertas de risco imediato.

* **Alta Sensibilidade (Recall Otimizado):** O algoritmo de Machine Learning foi calibrado para minimizar falsos negativos. É preferível que o RH faça uma entrevista de retenção preventiva do que ser pego de surpresa por uma demissão crítica.
* **Inteligência de Tempo Real:** Painel atualizado dinamicamente que cruza equidade salarial, demografia e tempo de casa.
* **Target List Acionável:** Uma "Fila de Prioridade" exportável que elenca os colaboradores ativos com probabilidade de fuga (Risco > 70%), cruzando as variáveis que mais impactam o negócio.

---

## 🧠 Arquitetura do Sistema (O Megazord)

O projeto é dividido em dois grandes núcleos: a **Esteira MLOps** (Extração, Tratamento e Treinamento) e o **Web App Executivo** (API e Dashboard).

### 🛠️ Tech Stack (O Cinto de Utilidades)
* **Backend & API:** `FastAPI`, `Uvicorn`, `Python 3.10+`
* **Engenharia de Dados & ML:** `Pandas`, `NumPy`, `Scikit-Learn` (Logistic Regression, StandardScaler), `Imbalanced-Learn` (SMOTE)
* **Banco de Dados (Conexão e CRUD):** `PostgreSQL` (DW de origem), `SQLite` (Gestão de Acessos do Dashboard), `SQLAlchemy`
* **Frontend Analytics:** `HTML5`, `CSS3` (Apple-inspired UI), `Vanilla JS`, `Plotly.js` (Visualização de Dados), `SheetJS` (Exportação Excel)

### 📂 Estrutura de Diretórios
```bash
📦 logistic_regression_arq
 ┣ 📂 Data
 ┃ ┣ 📂 Processed       # Checkpoints com dados limpos e features forjadas
 ┃ ┗ 📂 Raw             # Backup bruto extraído do DW
 ┣ 📂 Logs              # Matrizes de Confusão e auditoria de treinamento
 ┣ 📂 Models            # Cérebro da IA (Pipeline .pkl exportado)
 ┣ 📂 Src               # O Batalhão de Engenharia (Scripts Modulares)
 ┃ ┣ 📜 data_cleaning.py       # Tratamento de nulos e agrupamento de categorias
 ┃ ┣ 📜 data_extraction.py     # Conexão PostgreSQL e extração da OBT
 ┃ ┣ 📜 database.py            # Motor SQLAlchemy lendo o .env
 ┃ ┣ 📜 feature_engineering.py # Forja de variáveis (Idade, Meses de Casa)
 ┃ ┣ 📜 logger.py              # Sistema de Rastreamento
 ┃ ┣ 📜 ml_preprocessing.py    # Split, Prevenção de Leakage e StandardScaler
 ┃ ┗ 📜 train.py               # Orquestração do SMOTE + Regressão Logística
 ┣ 📂 static            # Ativos do Frontend
 ┃ ┣ 📜 index.html             # O Dashboard Corporativo (Bento Grid)
 ┃ ┣ 📜 script.js              # Lógica de renderização Plotly e consumo de API
 ┃ ┗ 📜 style.css              # Design System (Dark Mode, UI/UX)
 ┣ 📜 main.py           # Gatilho da esteira de MLOps
 ┣ 📜 server.py         # Servidor FastAPI (Rotas, CRUD, Servidor de Arquivos)
 ┣ 📜 requirements.txt  # Dependências do projeto
 ┗ 📜 .env              # Credenciais do Banco de Dados (NÃO VERSIONADO)
```
---

### **🔬 Exploratory Data Analysis (EDA Completa)**

O painel central não apresenta apenas números frios. Ele utiliza o Plotly.js em layout fluído para entregar 8 visualizações estratégicas:

1. **Equidade Salarial (Split Violin Plot)**: Compara a distribuição salarial entre ativos e evasões, dividido por gênero ou perfil.

2. **Sobrevivência (Boxplot)**: Indica o "mês crítico" em que os colaboradores tendem a pedir demissão.

3. **Maturidade x Salário (Scatter Plot)**: Análise bivariada mostrando os clusters de risco no tempo.

4. **Volume por Departamento (Bar Chart)**: Onde está o maior gargalo organizacional.

5. **Pirâmide Etária (Histogram)**: Distribuição demográfica de quem sai vs. quem fica.

6. **Perfil Comportamental (Donut Chart)**: Qual perfil psicológico tem menor adesão à cultura atual.

7. **Carga Familiar (Horizontal Bar)**: Correlação de retenção baseada na quantidade de dependentes.

8. **Matriz de Correlação (Heatmap)**: A prova matemática (Pearson) de quais variáveis impulsionam o turnover.

---
### **⚙️ Como Operar a Infraestrutura Localmente**

1. **Clonar e Instalar**

``` bash
git clone [https://github.com/Ignowsky/logistic_regression_arq.git](https://github.com/Ignowsky/logistic_regression_arq.git)
cd logistic_regression_arq
python -m venv venv
# Ative o venv (Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

2. **Configurar o Cofre (.env)**
Crie um arquivo .env na raiz do projeto com as credenciais do seu Data Warehouse PostgreSQL:

``` bash
DB_USER=seu_usuario
DB_PASS=sua_senha
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_SCHEMA=gestao_pessoas
```

3. **Rodando o Treinamento Inicial (MLOps)**
Para forçar a primeira extração, limpar os dados, balancear com SMOTE e treinar a Inteligência Artificial:

``` bash
python main.py
```
> (Confira os logs no terminal e os gráficos gerados na pasta /Logs)

4. **Subindo o Servidor Enterprise (Deploy Local)**
Ligue a API e o Dashboard:
````bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
````
>Acesse http://localhost:8000 no seu navegador.

- Acesso Padrão Admin:
  - Usuário: `admin_rh`
  - Senha: `123456`

---

### **🔄 Retreino Contínuo (Model Drift Prevention)**
A plataforma possui um módulo embutido de "Gestão de Sistema" (disponível apenas para Administradores logados). Com um único clique no botão "Retreinar IA", o servidor aciona o main.py em background, puxa dados novos do banco, reconstrói a inteligência matemática e atualiza a interface sem precisar derrubar o sistema.
