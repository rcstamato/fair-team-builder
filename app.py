import streamlit as st
import pandas as pd

from optimizer import processar

st.title("Fair Team Builder")
st.text("Esse app divide uma lista de jogadores em quartetos, visando manter o equilíbrio entre os times")

st.header("Passo 1: definir os jogadores e scores")

jogadores_df = pd.DataFrame(columns=["id", "nome", "score", "sexo"], data=[[1, "Fulano", 8, "m"], [2, "Beltrana", 10, "f"]])

st.write("Você pode subir um arquivo CSV com os jogadores, ou preencher diretamente a tabela abaixo")
st.download_button("Exemplo de arquivo", jogadores_df.to_csv(index=False), file_name="jogadores.csv")
jogadores_file = st.file_uploader("Upload")

if jogadores_file is not None:
    jogadores_df = pd.read_csv(jogadores_file)

jogadores_df.set_index("id", inplace=True)
modified_df = st.data_editor(jogadores_df, num_rows="dynamic")


st.header("Passo 2: definir restrições")
st.write("Defina aqui restrições do tipo 'Fulano não pode jogar com Cicrano'")

col1, col2, col3 = st.columns(3, vertical_alignment="center")

player_a = col1.selectbox("Jogador A", modified_df.index, format_func=lambda x: modified_df.loc[x, "nome"])

col2.html("<p align='center' style='margin-top: 2rem; height: 100%'>não pode jogar com</p>")

player_b = col3.selectbox("Jogador B", modified_df.index, format_func=lambda x: modified_df.loc[x, "nome"])

if "restricoes" not in st.session_state:
    restricoes = pd.DataFrame(columns=["id_a", "nome_a", "id_b", "nome_b"])
    st.session_state["restricoes"] = restricoes
else:
    restricoes = st.session_state["restricoes"]

with st.container(horizontal_alignment="right"):
    button_disabled = player_a == player_b
    if st.button("Adicionar restrição", icon=":material/arrow_downward:", disabled=button_disabled):
        restricoes.loc[len(restricoes)] = [player_a, modified_df.loc[player_a, "nome"], player_b, modified_df.loc[player_b, "nome"]]


st.dataframe(restricoes[["nome_a", "nome_b"]])

st.header("Passo 3: definir os times")

if st.button("Gerar times"):

    times, metricas = processar(jogadores_df, restricoes)

    st.write("Solução encontrada!\n")
    
    for idx, membros in enumerate(times, start=1):
        media = sum(n for _, n in membros) / len(membros)
        membros_nomes = [(jogadores_df[jogadores_df["id"]==id]["nome"].values[0], nota) for (id, nota) in membros]
        st.subheader(f"Time {idx}")
        membros_ids = [id for (id, nota) in membros]
        st.table(jogadores_df.set_index("id").loc[membros_ids].sort_values("score", ascending=False))
        # for membro in membros_nomes:
        #     st.write(f"{membro[0]} ({membro[1]})")

    # st.write("\nMétricas:")
    # st.write(f"  Média-alvo por time: {metricas['media_alvo_por_time']:.3f}")
    # st.write(f"  Somas por time: {metricas['somas_por_time']}")
    # st.write(f"  Desvios absolutos: {[round(v,3) for v in metricas['desvios_absolutos']]}")
    # st.write(f"  Range (max-min): {metricas['range']:.3f}")
    # st.write(f"  Valor da função objetivo: {metricas['objetivo']:.3f}")