import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

BASE = Path(__file__).parent / 'data'

st.set_page_config(page_title='REE Explorer', layout='wide')
st.title('Rare Earth Elements Explorer (Prototype)')

# Load CSVs
elements = pd.read_csv(BASE / 'elements.csv')
alloys = pd.read_csv(BASE / 'alloys.csv')
minerals = pd.read_csv(BASE / 'minerals.csv')
supply = pd.read_csv(BASE / 'supply.csv')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Elements','Alloys','Minerals Map','Market','Q&A'])

with tab1:
    st.header('Elements Explorer')
    q = st.text_input('Filter elements (name, application, property)...', '')
    if q:
        mask = elements.apply(lambda r: q.lower() in r.astype(str).str.lower().to_string(), axis=1)
        df = elements[mask]
    else:
        df = elements
    st.dataframe(df, use_container_width=True)
    if st.button('Show selected summary') and q:
        if not df.empty:
            r = df.iloc[0]
            st.markdown(f"**{r['element']} ({r['symbol']})**\n\n{r['notes']}")
        else:
            st.info('No matching element')

with tab2:
    st.header('Alloys & Compounds')
    q2 = st.text_input('Filter alloys (name, application, property)...', '')
    if q2:
        mask2 = alloys.apply(lambda r: q2.lower() in r.astype(str).str.lower().to_string(), axis=1)
        df2 = alloys[mask2]
    else:
        df2 = alloys
    st.dataframe(df2, use_container_width=True)
    if st.checkbox('Compare two alloys'):
        choices = st.multiselect('Select two alloys', list(alloys['alloy']), default=list(alloys['alloy'])[:2])
        if len(choices) == 2:
            a = alloys[alloys['alloy']==choices[0]].iloc[0]
            b = alloys[alloys['alloy']==choices[1]].iloc[0]
            comp = pd.DataFrame([a, b]).T
            st.dataframe(comp)
        else:
            st.info('Select exactly two alloys to compare')

with tab3:
    st.header('Mineral Occurrences Map')
    st.write('Interactive map of sample REE deposits (source: sample CSV)')
    fig = px.scatter_mapbox(minerals, lat='latitude', lon='longitude', hover_name='deposit_name',
                            hover_data=['key_REEs','grade_pct'], zoom=1, height=600)
    fig.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header('Market & Supply')
    elem = st.selectbox('Select element', supply['element'].unique())
    df_supply = supply[supply['element']==elem]
    fig2 = px.bar(df_supply, x='country', y='production_tonnes', color='country', title=f'Production of {elem} (sample data)')
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df_supply, use_container_width=True)

with tab5:
    st.header('Q&A Assistant (TF-IDF local)')
    # build mini-corpus from documents and CSV text
    docs = []
    meta = []
    # load text docs
    docs_path = BASE / 'documents'
    for fp in docs_path.glob('*.txt'):
        txt = fp.read_text()
        docs.append(txt)
        meta.append({'source': fp.name})
    # add short CSV-derived sentences
    for _, r in elements.iterrows():
        docs.append(f"{r['element']}: {r['key_applications']}. {r['notes']}")
        meta.append({'source': 'elements.csv'})
    for _, r in alloys.iterrows():
        docs.append(f"{r['alloy']}: {r['key_applications']}. {r['notes']}")
        meta.append({'source': 'alloys.csv'})

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(docs)

    user_q = st.text_input('Ask a question about REEs, alloys, or recycling...','Which REE is used in magnets?')
    topk = st.slider('Number of retrieved snippets', 1, 5, 3)
    if st.button('Get Answer'):
        qv = vectorizer.transform([user_q])
        sims = cosine_similarity(qv, X).flatten()
        idxs = sims.argsort()[::-1][:topk]
        st.subheader('Top evidence snippets')
        for i in idxs:
            st.write(f"- Source: {meta[i]['source']}")
            st.write(docs[i])
        # simple synthesized answer: combine top sentences
        import re
        synth = ' '.join([docs[i] for i in idxs])
        sentences = re.split(r'(?<=[.!?])\\s+', synth)
        answer = ' '.join(sentences[:4])
        st.markdown('**Answer (synthesized):**')
        st.write(answer)
