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
    const abas = document.querySelectorAll('.tab-content');
    abas.forEach(aba => aba.classList.remove('active-tab'));

    const botoes = document.querySelectorAll('.tab-btn');
    botoes.forEach(btn => btn.classList.remove('active'));

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
    try {
        const dep = document.getElementById('filtro-departamento').value;
        const res = await fetch(`/api/organizational_health?departamento=${dep}`);

        if (!res.ok) throw new Error("Erro na API do Backend");

        const data = await res.json();
        window.targetListData = data.target_list;

        // Atualiza KPIs Iniciais
        document.getElementById('kpi-hc').innerText = data.kpis.headcount || 0;
        document.getElementById('kpi-turnover').innerText = (data.kpis.taxa_turnover || 0) + "%";
        document.getElementById('kpi-evasoes').innerText = data.kpis.evasoes || 0;

        // Cores Apple S-Rank
        const corAtivos = 'rgba(52, 199, 89, 0.7)'; // Verde Sucesso
        const corEvasoes = 'rgba(255, 59, 48, 0.7)'; // Vermelho Alerta
        const corM = '#2a388f'; // Azul
        const corF = '#ff2d55'; // Rosa

        const layoutBase = { margin: { t: 30, b: 40 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' };

        // ------------------ ABA 1: VISÃO GERAL ------------------
// ------------------ ABA 1: VISÃO GERAL ------------------

        // 1. DEPARTAMENTOS (Barra Horizontal para ler todos os nomes sem amassar)
        // Criando um array ordenado para o maior ficar no topo
        let depsOrdenados = [];
        for(let i = 0; i < data.departamentos.nomes.length; i++) {
            depsOrdenados.push({
                nome: data.departamentos.nomes[i],
                evasao: data.departamentos.evasoes[i]
            });
        }
        // Ordena do menor pro maior (no Plotly 'h', o maior vai pro topo do gráfico)
        depsOrdenados.sort((a,b) => a.evasao - b.evasao);

        Plotly.newPlot('chart-departamentos', [{
            y: depsOrdenados.map(d => d.nome),
            x: depsOrdenados.map(d => d.evasao),
            type: 'bar',
            orientation: 'h', // Vira o gráfico de lado
            marker: { color: corM, opacity: 0.8 },
            hovertemplate: '<b>%{y}</b><br>Volume de Evasões: %{x}<extra></extra>'
        }], { ...layoutBase, margin: { l: 160, t: 30, b: 40 } }); // Margem 'l' maior para caber os nomes longos


        // 2. PERFIL COMPORTAMENTAL (Com a resenha completa no Hover/Tooltip)
        const descricoesPerfis = {
            'CE': 'Comunicador / Executor: Dinâmico, persuasivo, focado em ação e resultados rápidos.',
            'EC': 'Executor / Comunicador: Liderança enérgica, pragmático, direto e carismático.',
            'PA': 'Planejador / Analista: Organizado, metódico, focado em qualidade, processos e estabilidade.',
            'AP': 'Analista / Planejador: Detalhista, perfeccionista, focado em regras e alta precisão.',
            'CA': 'Comunicador / Analista: Sociável, mas bastante atento a detalhes e dados.',
            'AC': 'Analista / Comunicador: Preciso e focado em fatos, mas sabe articular e se comunicar.',
            'CP': 'Comunicador / Planejador: Empático, focado no bem-estar da equipe, mediação e harmonia.',
            'PC': 'Planejador / Comunicador: Paciente, excelente ouvinte e agregador nato.',
            'EA': 'Executor / Analista: Focado em metas brutas com alto rigor técnico e qualidade.',
            'AE': 'Analista / Executor: Estratégico, exigente, focado em eficiência e execução precisa.',
            'EP': 'Executor / Planejador: Prático, foca em executar com cadência, ritmo e constância.',
            'PE': 'Planejador / Executor: Consistente, focado em manter rotinas e entregas seguras.',
            'C': 'Comunicador: Extrovertido, focado em pessoas, networking e conexões.',
            'E': 'Executor: Competitivo, focado 100% em metas, desafios e velocidade.',
            'P': 'Planejador: Calmo, previsível, focado em processos e constância de longo prazo.',
            'A': 'Analista: Lógico, crítico, focado em exatidão e qualidade profunda.',
            'Não Mapeado': 'Colaborador ainda não realizou o assessment na Solides.',
            'OUTROS': 'Grupos menores agregados pela IA.'
        };

        // Mapeia os textos para injetar na "caixinha" do mouse
        const textosHover = data.perfil.nomes.map(p => descricoesPerfis[p] || 'Resumo não disponível.');

        Plotly.newPlot('chart-perfil', [{
            labels: data.perfil.nomes, // Mantém a sigla fora
            values: data.perfil.valores,
            type: 'pie',
            hole: 0.5,
            textinfo: 'label+percent',
            hovertext: textosHover, // Injeta o textão
            hovertemplate: '<b>Perfil %{label}</b><br>%{hovertext}<br><br>Evasões: %{value} (%{percent})<extra></extra>', // O formato da caixinha
            marker: { colors: [corM, corF, '#7a85e0', '#a8b0eb', '#c9cff5'] }
        }], layoutBase);

        // ------------------ ABA 2: DEMOGRAFIA (EDA) ------------------
        if (data.eda_avancada) {
            const eda = data.eda_avancada;

            // Filtros de status
            const sal_ativos = eda.salario.filter((_,i) => eda.status[i] === 0);
            const sal_evasoes = eda.salario.filter((_,i) => eda.status[i] === 1);
            const tempo_ativos = eda.tempo.filter((_,i) => eda.status[i] === 0);
            const tempo_evasoes = eda.tempo.filter((_,i) => eda.status[i] === 1);

            // Prevenção caso a idade falte no backend temporariamente
            const idades = eda.idade || Array(eda.salario.length).fill(30);
            const idade_ativos = idades.filter((_,i) => eda.status[i] === 0);
            const idade_evasoes = idades.filter((_,i) => eda.status[i] === 1);

            const generos_unicos = [...new Set(eda.genero)];
            const gen1 = generos_unicos[0] || 'Gen 1';
            const gen2 = generos_unicos[1] || 'Gen 2';

            const tempo_gen1 = eda.tempo.filter((_,i) => eda.genero[i] === gen1);
            const tempo_gen2 = eda.tempo.filter((_,i) => eda.genero[i] === gen2);

            const idade_gen1 = idades.filter((_,i) => eda.genero[i] === gen1);
            const idade_gen2 = idades.filter((_,i) => eda.genero[i] === gen2);

            // 1. KDE Salário
            Plotly.newPlot('chart-kde-salario', [
                { x: sal_ativos, type: 'histogram', histnorm: 'probability density', name: 'Ativos', marker: {color: corAtivos} },
                { x: sal_evasoes, type: 'histogram', histnorm: 'probability density', name: 'Evasões', marker: {color: corEvasoes} }
            ], { ...layoutBase, barmode: 'overlay', xaxis: {title: 'Salário (R$)'}, yaxis: {visible: false} });

            // 2. KDE Tempo por Gênero/Perfil
            Plotly.newPlot('chart-kde-tempo-genero', [
                { type: 'violin', x: tempo_gen1, name: gen1, line: {color: corM}, side: 'both', orientation: 'h' },
                { type: 'violin', x: tempo_gen2, name: gen2, line: {color: corF}, side: 'both', orientation: 'h' }
            ], { ...layoutBase, xaxis: {title: 'Meses de Casa'} });

            // 3. Boxplot Idade
            Plotly.newPlot('chart-box-idade-genero', [
                { y: idade_gen1, type: 'box', name: gen1, marker: {color: corM} },
                { y: idade_gen2, type: 'box', name: gen2, marker: {color: corF} }
            ], { ...layoutBase, yaxis: {title: 'Idade'} });

            // 4. Pie Gênero Evasão
            const evasao_por_genero = eda.genero.filter((_,i) => eda.status[i] === 1);
            const counts_genero = generos_unicos.map(g => evasao_por_genero.filter(x => x === g).length);
            Plotly.newPlot('chart-pie-genero-evasao', [{
                labels: generos_unicos,
                values: counts_genero,
                type: 'pie',
                hole: 0.5,
                marker: { colors: [corM, corF, '#a8b0eb'] }
            }], layoutBase);

            // 5. Scatter Maturidade
            Plotly.newPlot('chart-scatter-maturidade', [
                { x: tempo_ativos, y: sal_ativos, mode: 'markers', name: 'Ativos', marker: {color: corAtivos, size: 8} },
                { x: tempo_evasoes, y: sal_evasoes, mode: 'markers', name: 'Evasões', marker: {color: corEvasoes, symbol: 'diamond', size: 8} }
            ], { ...layoutBase, xaxis: {title: 'Maturidade (Meses)'}, yaxis: {title: 'Salário (R$)'} });

            // 6. Scatter Idade
            Plotly.newPlot('chart-scatter-idade', [
                { x: idade_ativos, y: sal_ativos, mode: 'markers', name: 'Ativos', marker: {color: corAtivos, size: 8} },
                { x: idade_evasoes, y: sal_evasoes, mode: 'markers', name: 'Evasões', marker: {color: corEvasoes, symbol: 'diamond', size: 8} }
            ], { ...layoutBase, xaxis: {title: 'Idade'}, yaxis: {title: 'Salário (R$)'} });
        }

    } catch (error) {
        console.error("Falha ao carregar dados:", error);
    }
}


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

async function dispararRetreino() {
    const btn = document.getElementById('btn-retrain');
    btn.innerHTML = '⚙️ Rodando Esteira S-Rank... Aguarde';
    btn.style.backgroundColor = '#ff9500';
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