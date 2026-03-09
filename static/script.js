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
    const corAtivos = '#34c759'; // Verde Sucesso
    const corEvasoes = '#ff3b30'; // Vermelho Alerta
    const corPrimaria = '#2a388f'; // Arq Azul

    // Gráfico 1: Evasões por Departamento
    Plotly.newPlot('chart-departamentos', [{
        x: data.departamentos.nomes,
        y: data.departamentos.evasoes,
        type: 'bar',
        marker: { color: corPrimaria, opacity: 0.8 }
    }], { margin: { t: 20 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' });

    // Gráfico 2: Equidade Salarial (Violin)
    Plotly.newPlot('chart-salario', [
        { type: 'violin', y: data.eda_avancada.salario.filter((_,i)=>data.eda_avancada.status[i]===0), name: 'Ativos', side: 'negative', line: {color: corAtivos} },
        { type: 'violin', y: data.eda_avancada.salario.filter((_,i)=>data.eda_avancada.status[i]===1), name: 'Evasões', side: 'positive', line: {color: corEvasoes} }
    ], { margin: { t: 20 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', violingap: 0, violinmode: 'overlay' });

    // 🔥 O NOVO GRÁFICO: Tempo Médio de Casa (Ativos vs Evasões)
    Plotly.newPlot('chart-tempo-medio', [{
        x: ['Colaboradores Ativos', 'Evasões (Saídas)'],
        y: [data.tempo_medio.ativos, data.tempo_medio.evasoes],
        type: 'bar',
        text: [data.tempo_medio.ativos + ' meses', data.tempo_medio.evasoes + ' meses'],
        textposition: 'auto',
        marker: { color: [corAtivos, corEvasoes], opacity: 0.9 }
    }], {
        margin: { t: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        yaxis: { title: 'Meses de Casa' }
    });

    // Gráfico 4: Dispersão Salário vs Maturidade
    Plotly.newPlot('chart-dispersao', [
        { x: data.eda_avancada.tempo.filter((_,i)=>data.eda_avancada.status[i]===0), y: data.eda_avancada.salario.filter((_,i)=>data.eda_avancada.status[i]===0), mode: 'markers', name: 'Ativos', marker: {color: 'rgba(52, 199, 89, 0.4)'} },
        { x: data.eda_avancada.tempo.filter((_,i)=>data.eda_avancada.status[i]===1), y: data.eda_avancada.salario.filter((_,i)=>data.eda_avancada.status[i]===1), mode: 'markers', name: 'Evasões', marker: {color: corEvasoes, symbol: 'diamond'} }
    ], { margin: { t: 20 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' });

    // Gráfico 5: Evasões por Perfil Comportamental
    Plotly.newPlot('chart-perfil', [{
        labels: data.perfil.nomes,
        values: data.perfil.valores,
        type: 'pie',
        hole: 0.4,
        marker: { colors: [corPrimaria, '#4c5ad6', '#7a85e0', '#a8b0eb', '#d6dcf5'] }
    }], { margin: { t: 20, b:20 }, paper_bgcolor: 'rgba(0,0,0,0)' });
}

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
        const res = await fetch('/api/retrain', { method: 'POST' });
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