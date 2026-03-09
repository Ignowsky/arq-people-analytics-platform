let token = localStorage.getItem('token');
let userRole = localStorage.getItem('role');

// Verifica login no carregamento
if (!token) {
    document.getElementById('login-overlay').style.display = 'flex';
} else {
    iniciarSistema();
}

async function fazerLogin() {
    const u = document.getElementById('username').value;
    const p = document.getElementById('password').value;
    try {
        const res = await fetch('/api/auth', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: u, password: p})
        });
        if (res.ok) {
            const data = await res.json();
            localStorage.setItem('token', data.token);
            localStorage.setItem('role', data.role);
            token = data.token;
            userRole = data.role;
            document.getElementById('login-overlay').style.display = 'none';
            iniciarSistema();
        } else {
            document.getElementById('login-error').style.display = 'block';
        }
    } catch (e) {
        console.error(e);
    }
}

function sair() {
    localStorage.clear();
    location.reload();
}

function iniciarSistema() {
    document.getElementById('main-dashboard').style.display = 'block';
    if (userRole === 'Administrador') {
        document.getElementById('btn-admin-tab').style.display = 'block';
        carregarUsuarios();
    }
    carregarDepartamentos();
    carregarDados();
}

// --- CONTROLE DE ABAS ---
function mostrarAba(abaId, elementoBotao) {
    // Esconde todas as abas
    const abas = document.querySelectorAll('.tab-content');
    abas.forEach(aba => aba.classList.remove('active-tab'));

    // Remove o estilo ativo de todos os botões
    const botoes = document.querySelectorAll('.tab-btn');
    botoes.forEach(btn => btn.classList.remove('active'));

    // Mostra a aba selecionada e ativa o botão
    document.getElementById(abaId).classList.add('active-tab');
    elementoBotao.classList.add('active');
}

// --- CARREGAMENTO DE DADOS ---
async function carregarDepartamentos() {
    const res = await fetch('/api/departments');
    const deps = await res.json();
    const select = document.getElementById('filtro-departamento');
    deps.forEach(d => {
        let opt = document.createElement('option');
        opt.value = d; opt.innerHTML = d;
        select.appendChild(opt);
    });
}

async function carregarDados() {
    const dep = document.getElementById('filtro-departamento').value;
    const res = await fetch(`/api/organizational_health?departamento=${dep}`);
    const data = await res.json();
    window.targetListData = data.target_list;

    // Atualiza KPIs
    document.getElementById('kpi-hc').innerText = data.kpis.headcount;
    document.getElementById('kpi-turnover').innerText = data.kpis.taxa_turnover + "%";
    document.getElementById('kpi-evasoes').innerText = data.kpis.evasoes;

    // Cores Apple S-Rank
// Cores Apple S-Rank
    const corAtivos = 'rgba(52, 199, 89, 0.7)'; // Verde Sucesso Transparente
    const corEvasoes = 'rgba(255, 59, 48, 0.7)'; // Vermelho Alerta Transparente
    const corM = '#2a388f'; // Azul Masculino/Geral
    const corF = '#ff2d55'; // Rosa Apple Feminino

    // PREPARAÇÃO DOS DADOS (Fatiando o JSON pra não repetir código)
    const eda = data.eda_avancada;

    // Arrays filtrados por Status
    const sal_ativos = eda.salario.filter((_,i) => eda.status[i] === 0);
    const sal_evasoes = eda.salario.filter((_,i) => eda.status[i] === 1);

    const tempo_ativos = eda.tempo.filter((_,i) => eda.status[i] === 0);
    const tempo_evasoes = eda.tempo.filter((_,i) => eda.status[i] === 1);

    const idade_ativos = eda.idade.filter((_,i) => eda.status[i] === 0);
    const idade_evasoes = eda.idade.filter((_,i) => eda.status[i] === 1);

    // Arrays filtrados por Gênero (Assumindo M e F ou Perfil se Gênero não existir)
    // Se a sua base tiver "Masculino/Feminino", ajuste aqui, mas o filter genérico pega o que vier
    const generos_unicos = [...new Set(eda.genero)];
    const gen1 = generos_unicos[0];
    const gen2 = generos_unicos[1] || 'Outro';

    const tempo_gen1 = eda.tempo.filter((_,i) => eda.genero[i] === gen1);
    const tempo_gen2 = eda.tempo.filter((_,i) => eda.genero[i] === gen2);

    const idade_gen1 = eda.idade.filter((_,i) => eda.genero[i] === gen1);
    const idade_gen2 = eda.idade.filter((_,i) => eda.genero[i] === gen2);

    // Layout padrão pra manter o fundo transparente e limpo (Padrão Apple)
    const layoutKDE = { margin: { t: 20 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', showlegend: true };

    // -------------------------------------------------------------
    // 1. KDE: Distribuição Salarial (Ativos vs Evasões) - Sem o "pulmão murcho"
    // Usamos Histograma com densidade de probabilidade e formato de curva (KDE real)
    // -------------------------------------------------------------
    Plotly.newPlot('chart-kde-salario', [
        { x: sal_ativos, type: 'histogram', histnorm: 'probability density', name: 'Ativos', marker: {color: corAtivos} },
        { x: sal_evasoes, type: 'histogram', histnorm: 'probability density', name: 'Evasões', marker: {color: corEvasoes} }
    ], { ...layoutKDE, barmode: 'overlay', xaxis: {title: 'Salário (R$)'}, yaxis: {visible: false} });

    // -------------------------------------------------------------
    // 2. KDE: Tempo de Casa por Gênero (Violin Plot completo, deitado)
    // -------------------------------------------------------------
    Plotly.newPlot('chart-kde-tempo-genero', [
        { type: 'violin', x: tempo_gen1, name: gen1, line: {color: corM}, side: 'both', orientation: 'h', meanline: {visible: true} },
        { type: 'violin', x: tempo_gen2, name: gen2, line: {color: corF}, side: 'both', orientation: 'h', meanline: {visible: true} }
    ], { ...layoutKDE, xaxis: {title: 'Meses de Casa'} });

    // -------------------------------------------------------------
    // 3. Distribuição de Idade por Gênero (Box Plot)
    // -------------------------------------------------------------
    Plotly.newPlot('chart-box-idade-genero', [
        { y: idade_gen1, type: 'box', name: gen1, marker: {color: corM}, boxpoints: 'outliers' },
        { y: idade_gen2, type: 'box', name: gen2, marker: {color: corF}, boxpoints: 'outliers' }
    ], { ...layoutKDE, yaxis: {title: 'Idade'} });

    // -------------------------------------------------------------
    // 4. Composição de Gênero nas Evasões (Gráfico de Rosca)
    // -------------------------------------------------------------
    const evasao_por_genero = eda.genero.filter((_,i) => eda.status[i] === 1);
    const counts_genero = generos_unicos.map(g => evasao_por_genero.filter(x => x === g).length);

    Plotly.newPlot('chart-pie-genero-evasao', [{
        labels: generos_unicos,
        values: counts_genero,
        type: 'pie',
        hole: 0.5,
        marker: { colors: [corM, corF, '#a8b0eb'] }
    }], layoutKDE);

    // -------------------------------------------------------------
    // 5. Bivariada: Dispersão de Salário vs Maturidade (Meses de Casa)
    // -------------------------------------------------------------
    Plotly.newPlot('chart-scatter-maturidade', [
        { x: tempo_ativos, y: sal_ativos, mode: 'markers', name: 'Ativos', marker: {color: corAtivos, size: 8, line: {width: 1, color: 'white'}} },
        { x: tempo_evasoes, y: sal_evasoes, mode: 'markers', name: 'Evasões', marker: {color: corEvasoes, symbol: 'diamond', size: 8, line: {width: 1, color: 'white'}} }
    ], { ...layoutKDE, xaxis: {title: 'Maturidade (Meses de Casa)'}, yaxis: {title: 'Salário (R$)'} });

    // -------------------------------------------------------------
    // 6. Bivariada: Dispersão de Salário vs Idade
    // -------------------------------------------------------------
    Plotly.newPlot('chart-scatter-idade', [
        { x: idade_ativos, y: sal_ativos, mode: 'markers', name: 'Ativos', marker: {color: corAtivos, size: 8, line: {width: 1, color: 'white'}} },
        { x: idade_evasoes, y: sal_evasoes, mode: 'markers', name: 'Evasões', marker: {color: corEvasoes, symbol: 'diamond', size: 8, line: {width: 1, color: 'white'}} }
    ], { ...layoutKDE, xaxis: {title: 'Idade'}, yaxis: {title: 'Salário (R$)'} });

// --- TARGET LIST ---
function baixarTargetList() {
    if(!window.targetListData || window.targetListData.length === 0) return alert("Sem dados para exportar.");
    let csv = "Colaborador_SK,Departamento,Perfil,Risco_IA_Evasao(%)\n";
    window.targetListData.forEach(r => {
        csv += `${r.colaborador_sk},${r.departamento_nome_api},${r.perfil_comportamental},${r.risco}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'Target_List_Risco_Evasao.csv';
    a.click();
}

// --- CRUD DE USUÁRIOS (COM E-MAIL) ---
async function carregarUsuarios() {
    const res = await fetch('/api/users');
    const users = await res.json();
    const tbody = document.getElementById('tabela-usuarios');
    tbody.innerHTML = '';
    users.forEach(u => {
        tbody.innerHTML += `<tr>
            <td>${u.id}</td>
            <td><strong>${u.username}</strong></td>
            <td>${u.email}</td>
            <td><span style="background: ${u.role==='Administrador' ? '#ffe5e5' : '#e5f0ff'}; color: ${u.role==='Administrador' ? '#ff3b30' : '#0066cc'}; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600;">${u.role}</span></td>
            <td><button class="btn-danger-text" onclick="deletarUsuario(${u.id})">Remover</button></td>
        </tr>`;
    });
}

async function adicionarUsuario() {
    const u = document.getElementById('new-user').value;
    const e = document.getElementById('new-email').value;
    const p = document.getElementById('new-pass').value;
    const r = document.getElementById('new-role').value;
    const res = await fetch('/api/users', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({username: u, email: e, password: p, role: r})
    });
    if(res.ok) {
        alert("Usuário S-Rank adicionado na base!");
        carregarUsuarios();
    } else {
        alert("Erro: Usuário ou E-mail já existem.");
    }
}

async function deletarUsuario(id) {
    if(confirm("Expulsar esse usuário da base?")) {
        await fetch(`/api/users/${id}`, {method: 'DELETE'});
        carregarUsuarios();
    }
}

// --- RETREINO (GATILHO DO MLOPS) ---
async function dispararRetreino() {
    const btn = document.getElementById('btn-retrain');
    btn.innerHTML = '⚙️ Rodando Esteira S-Rank... Aguarde';
    btn.style.backgroundColor = '#ff9500'; // Laranja Apple (Alerta/Processando)
    btn.disabled = true;

    try {
        const res = await fetch('/api/retrain', {method: 'POST'});
        if (res.ok) {
            alert('Sucesso! Megazord retreinado. Novos padrões detectados.');
            carregarDados();
        } else {
            const err = await res.json();
            alert('Falha na esteira. Verifique os logs do servidor.\nMotivo: ' + err.detail);
        }
    } catch (error) {
        alert('Erro de rede ao tentar acionar o retreino.');
    } finally {
        btn.innerHTML = '🔄 Retreinar IA (Atualizar Base)';
        btn.style.backgroundColor = 'var(--success-green)';
        btn.disabled = false;
        }
    }
}