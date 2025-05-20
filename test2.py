#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(layout="centered")

# --- Chargement des données ---
data = pd.read_csv("ab_data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])

# --- Titre du dashboard ---
st.title("Analyse A/B Test : Ancienne vs Nouvelle Page")

st.image("ab_testing.jpg", use_column_width=True)

# --- KPI Section 1 : Vue Générale ---
st.header("1️⃣  Vue d'ensemble des données")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Nombre total de visiteurs", len(data))

with col2:
    visitors_per_group = data['group'].value_counts()
    st.metric("Contrôle", visitors_per_group.get('control', 0))

with col3:
    st.metric("Traitement", visitors_per_group.get('treatment', 0))

# Taux de conversion global
conversion_rate = data['converted'].mean()
st.write(f"\n✅ **Taux de conversion global** : {conversion_rate:.2%}")

# --- KPI par groupe ---
st.subheader("Taux de conversion par groupe")
conversion_by_group = data.groupby('group')['converted'].mean().reset_index()
st.dataframe(conversion_by_group)

# --- Bar chart conversions ---
st.subheader(" 2️⃣ Visualisation : Taux de conversion par groupe")
bar_chart = alt.Chart(conversion_by_group).mark_bar().encode(
    x=alt.X('group:N', title='Groupe'),
    y=alt.Y('converted:Q', title='Taux de conversion'),
    color='group'
)
st.altair_chart(bar_chart, use_container_width=True)

# Visualisation 3: Heatmap groupe vs landing_page
st.subheader("3️⃣ Répartition Groupe vs Page")
cross_tab = pd.crosstab(data['group'], data['landing_page'])
fig_heatmap = plt.figure(figsize=(6, 4))  # Make the figure smaller
sns.heatmap(cross_tab, annot=True, fmt='d', cmap="Blues")
st.pyplot(fig_heatmap)

# Visualisation 4 : Nombre d'utilisateurs par groupe
st.subheader("4️⃣ Nombre d'utilisateurs par groupe")
group_counts = data['group'].value_counts()
st.bar_chart(group_counts)

st.subheader("5️⃣ Conversion dans le temps par groupe")
# Conversion du timestamp et extraction du jour de la semaine
data['day_of_week'] = data['timestamp'].dt.day_name()
# Filtrage des données pour les groupes A et B
group_a = data[data['group'] == 'control']
group_b = data[data['group'] == 'treatment']

# Calcul de la conversion quotidienne pour le groupe A
daily_conversion_a = group_a.groupby(group_a['timestamp'].dt.date)['converted'].mean()

# Calcul de la conversion quotidienne pour le groupe B
daily_conversion_b = group_b.groupby(group_b['timestamp'].dt.date)['converted'].mean()

# Affichage des graphiques côte à côte
col1, col2 = st.columns(2)

with col1:
    st.subheader("Groupe A")
    st.line_chart(daily_conversion_a)

with col2:
    st.subheader("Groupe B")
    st.line_chart(daily_conversion_b)

# Visualisation 5 : Taux de conversion par jour de la semaine
st.subheader("6️⃣ Conversion par jour de la semaine")

# Calcul du taux de conversion par jour de la semaine pour le groupe A
conversion_by_day_a = group_a.groupby('day_of_week')['converted'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Calcul du taux de conversion par jour de la semaine pour le groupe B
conversion_by_day_b = group_b.groupby('day_of_week')['converted'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Affichage des graphiques côte à côte
col1, col2 = st.columns(2)

with col1:
    st.subheader("Groupe A")
    st.bar_chart(conversion_by_day_a)

with col2:
    st.subheader("Groupe B")
    st.bar_chart(conversion_by_day_b)

df2 = data[((data['group'] == 'treatment') == (data['landing_page'] == 'new_page')) == True]    
df2.drop_duplicates(keep='first')

# --- Question interactive avant la Partie 2 ---
st.subheader("❓Avant d'aller plus loin")

with st.expander("🧠 Question : Peut-on conclure que la nouvelle page est meilleure ?"):
    st.markdown("**Est-il raisonnable de tirer une conclusion maintenant ? Pourquoi ?**")
    
    q1 = st.checkbox("🔍 Oui, les premières données suggèrent une différence entre les groupes.")
    q2 = st.checkbox("⚠️ Non, il est encore trop tôt pour conclure avec certitude.")
    q3 = st.checkbox("🤔 Je ne suis pas sûr, il faudrait analyser plus en profondeur.")

    if q1 or q2 or q3:
        st.markdown("---")
        st.markdown("### 🧾 Réflexion attendue :")
        st.write("- À ce stade, nous avons seulement des indicateurs visuels.")
        st.write("- Pour **confirmer** une différence réelle entre les groupes, **un test statistique est nécessaire**.")
        st.write("- C’est ce que nous allons faire dans la **Partie 2 : Test d’hypothèse**.")



# --- Partie 2 : Simulation sous l'hypothèse nulle ---
st.header("2. Test d'hypothèse par simulation")

# Paramètres de base
p_null = df2['converted'].mean()
n_new = df2.query("group == 'treatment'").shape[0]
n_old = df2.query("group == 'control'").shape[0]

# Simulation de 10 000 expériences
np.random.seed(42)
diffs = []
for _ in range(10000):
    new_sample = np.random.binomial(1, p_null, n_new)
    old_sample = np.random.binomial(1, p_null, n_old)
    diffs.append(new_sample.mean() - old_sample.mean())

diffs = np.array(diffs)

# Différence observée
obs_diff = df2.query("group == 'treatment'")['converted'].mean() - \
           df2.query("group == 'control'")['converted'].mean()

# p-value
p_value = (diffs > obs_diff).mean()

# KPI Résultats
col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Conversion Traitement", f"{df2.query('group == \"treatment\"')['converted'].mean():.2%}")
with col5:
    st.metric("Conversion Contrôle", f"{df2.query('group == \"control\"')['converted'].mean():.2%}")
with col6:
    st.metric("p-value", f"{p_value:.4f}")

# --- Histogramme des différences simulées ---
st.subheader("Distribution simulée des différences")
diffs_df = pd.DataFrame({'diffs': diffs})
chart = alt.Chart(diffs_df).mark_bar().encode(
    alt.X("diffs", bin=alt.Bin(maxbins=100), title="Différences simulées"),
    y='count()'
).properties(title="Distribution sous H₀")

line = alt.Chart(pd.DataFrame({'obs_diff': [obs_diff]})).mark_rule(color='red').encode(
    x='obs_diff'
)

st.altair_chart(chart + line, use_container_width=True)

# --- Interprétation ---
st.markdown("""
### 🎯 Conclusion
- La différence observée entre les taux de conversion est : **{0:.4f}**.
- La p-value est de **{1:.4f}**.
- Cela signifie que {2}
""".format(
    obs_diff,
    p_value,
    "cette différence est significative, la nouvelle page est probablement meilleure." if p_value < 0.05 else "la différence peut s'expliquer par le hasard. Nous ne rejetons pas l'hypothèse nulle."
))


