import streamlit as st
import pandas as pd
from sklearn.metrics import f1_score
from firebase_admin import firestore, credentials
import firebase_admin
import datetime
import time
import random

# --- Configurações Iniciais ---
COLECAO_PRINCIPAL = "competicao_ml_2025_1"
FUSO_HORARIO_AJUSTE = datetime.timedelta(hours=3)

# --- Inicialização do Firebase ---
def inicializar_firebase():
    """Inicializa o Firebase usando as credenciais do Streamlit Secrets."""
    try:
        # Tenta obter o app para verificar se já foi inicializado
        firebase_admin.get_app()
    except ValueError:
        # Se não foi inicializado, inicializa agora
        if 'firebase' in st.secrets:
            cred_dict = {
                "type": st.secrets.firebase.type,
                "project_id": st.secrets.firebase.project_id,
                "private_key_id": st.secrets.firebase.private_key_id,
                "private_key": st.secrets.firebase.private_key.replace('\\n', '\n'),
                "client_email": st.secrets.firebase.client_email,
                "client_id": st.secrets.firebase.client_id,
                "auth_uri": st.secrets.firebase.auth_uri,
                "token_uri": st.secrets.firebase.token_uri,
                "auth_provider_x509_cert_url": st.secrets.firebase.auth_provider_x509_cert_url,
                "client_x509_cert_url": st.secrets.firebase.client_x509_cert_url,
                "universe_domain": st.secrets.firebase.universe_domain
            }
            cred = credentials.Certificate(cred_dict)
        else:
            # Fallback para desenvolvimento local se o secrets não estiver disponível
            # Coloque seu arquivo 'firebase-key.json' no mesmo diretório
            cred = credentials.Certificate("firebase-key.json")
        
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = inicializar_firebase()

# --- Funções do Aplicativo ---

def validar_submissao(submissao_df, gabarito_vetor):
    """Valida o arquivo de submissao do usuario."""
    if submissao_df.shape[0] != len(gabarito_vetor):
        st.error(f"Erro: O número de linhas ({submissao_df.shape[0]}) é diferente do esperado ({len(gabarito_vetor)}).")
        return False, None
    
    if submissao_df.shape[1] != 1:
        st.error(f"Erro: O arquivo deve conter apenas uma coluna com as previsões.")
        return False, None
        
    try:
        # Garante que a coluna de previsão seja mapeada para o mesmo formato numérico
        previsao_mapeada = submissao_df.iloc[:, 0].map({'mau pagador': 1, 'bom pagador': 0})
        if previsao_mapeada.isnull().any():
            st.warning("Atenção: Alguns valores não são 'mau pagador' ou 'bom pagador' e serão ignorados no cálculo.")
            previsao_mapeada = previsao_mapeada.dropna()
        return True, previsao_mapeada
    except Exception as e:
        st.error(f"Erro ao processar a coluna de previsão: {e}")
        return False, None


def salvar_submissao(nome, f1, data_envio, descricao_modelo):
    """Salva os dados da submissao no Firestore."""
    doc_ref = db.collection(COLECAO_PRINCIPAL).document()
    doc_ref.set({
        'nome': nome,
        'f1_score': f1,
        'data_envio': data_envio,
        'descricao_modelo': descricao_modelo
    })

@st.cache_data(ttl=5) # Cache de 5 minutos
def carregar_submissoes():
    """Carrega todas as submissoes do Firestore."""
    submissoes = []
    docs = db.collection(COLECAO_PRINCIPAL).stream()
    for doc in docs:
        submissoes.append(doc.to_dict())
    
    df = pd.DataFrame(submissoes) if submissoes else pd.DataFrame()
    if not df.empty:
        df['data_envio'] = pd.to_datetime(df['data_envio'])
    return df

# --- Funções de Diálogo ---
@st.dialog("Resultado da Submissão")
def mostrar_dialog_resultado(f1_score):
    """Exibe um pop-up com o resultado do F1-Score."""
    st.markdown(f"## ✅ Submissão Enviada com Sucesso!")
    st.markdown(f"### Seu F1-Score é: **{f1_score:.4f}**")
    if st.button("OK", use_container_width=True):
        st.rerun()

# --- Interface do Streamlit ---

# Configura a página do Streamlit com título, ícone e layout amplo para melhor visualização
st.set_page_config(
    page_title = "Competição de Machine Learning 2025/1 - Ranking Oficial",
    page_icon = "🏆",
    layout = "centered",
    initial_sidebar_state = "collapsed"
)

# Adiciona a imagem de capa aleatória
imagens_capa = [f"https://raw.githubusercontent.com/ricardorocha86/Datasets/refs/heads/master/imagens_capa_competicao/Google_AI_Studio({i}).png" for i in range(1, 6)]
imagem_selecionada = random.choice(imagens_capa)
st.image(imagem_selecionada, use_container_width=True)

st.markdown("# **Competição de ML 🏆**")
st.caption("Envie sua previsão e veja sua posição no ranking!")

st.divider() 
# Carrega os dados de referência e submissões existentes
submissoes_df = carregar_submissoes()
nomes_existentes = sorted(submissoes_df['nome'].unique()) if not submissoes_df.empty else []

# Carrega o gabarito a partir dos secrets
try:
    gabarito_vetor = st.secrets.gabarito_vetor
except (AttributeError, FileNotFoundError):
    st.error("Erro fatal: O `gabarito_vetor` não foi encontrado no arquivo `secrets.toml`. Por favor, configure-o antes de continuar.")
    st.stop()


# --- Lógica de Seleção de Participante (Fora do Formulário) ---
eh_primeira_submissao = 'Sim'
if nomes_existentes:
    eh_primeira_submissao = st.radio(
        "Esta é a sua primeira submissão?",
        ('Sim', 'Não'),
        horizontal=True,
        key='tipo_submissao' # Adiciona uma chave para manter o estado
    )

# --- Seção de Submissão ---
nome_participante = None
if eh_primeira_submissao == 'Sim':
    if not nomes_existentes:
            st.info("👋 Bem-vindo! Para participar, insira seu nome abaixo.")
    nome_participante = st.text_input("Nome da equipe", max_chars=50, placeholder="Equipe Rocket")
else: # eh_primeira_submissao == 'Não'
    nome_participante = st.selectbox("Selecione seu nome", options=nomes_existentes)

descricao_modelo = st.text_input("Descrição do Modelo Utilizado", max_chars=100, placeholder="RandomForest Tunado v2")
arquivo_submetido = st.file_uploader("Selecione seu arquivo de submissão (.csv)", type=["csv"])

# O botão só é habilitado se todos os campos estiverem preenchidos
botao_desabilitado = not (nome_participante and descricao_modelo and arquivo_submetido)
enviado = st.button("Enviar Submissão", disabled=botao_desabilitado, use_container_width=True, type='primary')

if enviado:
    # 1. Barra de progresso falsa
    progress_bar = st.progress(0, text="Analisando sua submissão...")
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1, text=f"Calculando F1-Score... {i+1}%")
    
    submissao_df = pd.read_csv(arquivo_submetido)
    valido, previsao_mapeada = validar_submissao(submissao_df, gabarito_vetor)
    
    if valido:
        f1 = f1_score(gabarito_vetor, previsao_mapeada, pos_label=1)
        data_envio = datetime.datetime.now(datetime.timezone.utc)
        salvar_submissao(nome_participante, f1, data_envio, descricao_modelo)
        
        progress_bar.empty()
        st.balloons()
        
        # Chama a função decorada para exibir o pop-up
        mostrar_dialog_resultado(f1)
        
    else:
        progress_bar.empty()
        # O erro já é exibido pela função validar_submissao

st.divider() 

# --- Seção de Resultados ---
if not submissoes_df.empty:
    col1, col2 = st.columns([3, 2.1], border=True, gap = 'small')

    with col1:
        st.subheader("📜 Histórico de Submissões")
        historico_df = submissoes_df.sort_values(by="data_envio", ascending=False)
        
        # Garante que a coluna de descrição exista e preenche vazios
        if 'descricao_modelo' not in historico_df.columns:
            historico_df['descricao_modelo'] = ''
        historico_df['descricao_modelo'] = historico_df['descricao_modelo'].fillna('')

        # Formata colunas para exibição
        historico_df_display = historico_df.copy()
        historico_df_display['data_envio'] = (historico_df_display['data_envio'] - FUSO_HORARIO_AJUSTE).dt.strftime('%d/%m/%y - %H:%M')
        historico_df_display['f1_score'] = historico_df_display['f1_score'].map('{:.4f}'.format)
        
        st.dataframe(
            historico_df_display[['data_envio', 'nome', 'descricao_modelo', 'f1_score']].rename(
                columns={'data_envio': 'Data de Envio', 'nome': 'Nome', 'descricao_modelo': 'Modelo', 'f1_score': 'F1-Score'}
            ),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.subheader("🏆 Ranking Oficial")
        ranking_df = submissoes_df.loc[submissoes_df.groupby('nome')['f1_score'].idxmax()]
        ranking_df = ranking_df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        
        def atribuir_medalha(rank):
            if rank == 0: return '🥇'
            if rank == 1: return '🥈'
            if rank == 2: return '🥉'
            return ''
        
        ranking_df['Posição'] = [f"{i+1}º {atribuir_medalha(i)}" for i in ranking_df.index]
        ranking_df['f1_score'] = ranking_df['f1_score'].map('{:.4f}'.format)
        
        ranking_display = ranking_df[['Posição', 'nome', 'f1_score']].rename(columns={'nome': 'Nome', 'f1_score': 'F1-Score'})
        
        st.dataframe(
            ranking_display,
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("Aguardando a primeira submissão para exibir os resultados...") 