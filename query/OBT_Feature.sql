
CREATE OR REPLACE VIEW "analytics".vw_obt_turnover_lr AS
WITH
    -- 1° Bloco: Base de Dados Pura
    base_Colaboradores AS (
        SELECT
            -- Informações Pessoais Gerais
            colaborador_sk,
            data_nascimento,
            genero,
            estado_civil,
            escolaridade,
            cep,
            bairro,
            cidade,
            estado,

            -- Informações Corporativas
            data_admissao,
            data_demissao,
            nivel_hierarquico,
            departamento_nome_api,
            cargo_nome_api,
            turno_trabalho,
            tipo_contrato,
            salario_api, -- ⬅️ O JUTSU COMEÇA AQUI: Trazendo o salário padrão do cadastro

            -- Target que iremos buscar
            CASE
                WHEN decisao_demissao ILIKE 'pediu%' THEN 1
                ELSE 0
                END as target_pediu_demissao
        FROM "FOPAG".dim_colaboradores
        WHERE
            data_admissao IS NOT NULL AND
            nome_completo NOT ILIKE 'Lidiane%' AND
            (tipo_contrato != 'Sócio' OR tipo_contrato IS NULL) AND
          -- Para não sujar a OBT capturamos só quem pediu demissão
            (decisao_demissao ILIKE 'pediu%' OR data_demissao IS NULL)
    ),
    -- 2° Bloco: Perfil Comportalmental (Utilizando o testo mais recente)
    ultimo_profiler AS (
        SELECT DISTINCT ON (colaborador_sk)
            colaborador_sk,
            perfil_comportamental
        FROM "FOPAG".dim_profiler
        ORDER BY colaborador_sk,
                 data_teste DESC
    ),
    -- 3° Bloco: Contagem de Dependentes (Contamos quantos dependentes o colaborador tem)
    contagem_dependentes AS (
        SELECT colaborador_sk,
               COUNT(dependente_id) AS qtd_dependentes
        FROM "FOPAG".dim_dependentes
        GROUP BY colaborador_sk
    ),
    -- 4° Bloco: Calculo financeiro ( Considerando o último contracheque)
    ultima_folha AS (
        SELECT DISTINCT ON (colaborador_sk)
            colaborador_sk,
            salario_contratual,
            valor_liquido
        FROM "FOPAG".fato_folha_consolidada
        ORDER BY colaborador_sk,
                 competencia DESC
    )

-- =====================================
-- Montagem da Query (Cruzamentos Finais)
-- =====================================

SELECT
    -- Expandindo as colunas em vez de bC.* para manter a OBT limpa
    bC.colaborador_sk,
    bC.data_nascimento,
    bC.genero,
    bC.estado_civil,
    bC.escolaridade,
    bC.cep,
    bC.bairro,
    bC.cidade,
    bC.estado,
    bC.data_admissao,
    bC.data_demissao,
    bC.nivel_hierarquico,
    bC.departamento_nome_api,
    bC.cargo_nome_api,
    bC.turno_trabalho,
    bC.tipo_contrato,
    bC.target_pediu_demissao,

    COALESCE(cd.qtd_dependentes, 0) AS qtd_dependentes,
    up.perfil_comportamental,

    -- ⬅️ O GOLPE FINAL: Se a folha for nula, ele puxa o salario_api do cadastro como backup
    COALESCE(uf.salario_contratual, bC.salario_api) AS salario_contratual,
    uf.valor_liquido

FROM base_Colaboradores bC
         LEFT JOIN contagem_dependentes cd ON bC.colaborador_sk = cd.colaborador_sk
         LEFT JOIN ultimo_profiler up on bC.colaborador_sk = up.colaborador_sk
         LEFT JOIN ultima_folha uf on bC.colaborador_sk = uf.colaborador_sk;



