# Importing libraries
import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

def run():
        # Write title
        st.title("ONLINE GAMING BEHAVIOUR - OVERVIEW")

        # Load and encode the gif
        file_ = open("opus.gif", "rb")
        contents = file_.read()
        data_url = "data:image/gif;base64," + base64.b64encode(contents).decode("utf-8")

        # Display the gif using HTML
        st.markdown(
                f'<img src="{data_url}" alt="gif" style="width:100%;" />',
                unsafe_allow_html=True
        )

        # Write project description
        st.write('#####')
        st.header('Description')

        st.markdown('''In the fast-paced gaming industry of 2025, tracking playtime alone isn’t enough. To grow sustainably, game studios must understand how players engage and how it affects retention and monetization. 
                    This project predicts player Engagement Levels—Low, Medium, or High—using behavioral and demographic data. The goal is to uncover insights that help personalize player experiences and support smarter, data-driven business strategies.
                    ''')

        # Load the dataset
        df = pd.read_csv('P1M2_rd_ladityarsa_ilyankusuma.csv')
        st.header('Dataset')

        # Show the dataset as df
        st.dataframe(df)

        # Visualization with exploratory data analysis
        st.header('Exploratory Data Analysis')

        # Plot 1
        st.subheader('1. Target Classes Distribution')
        st.write('''Since the goal is to provide business recommendations based on predicted engagement levels, 
                 it’s important to first examine the distribution of the EngagementLevel column to see whether the classes are balanced or not.''')

        # Calculate normalized class distributions for the dataset's target
        dist = df['EngagementLevel'].value_counts(normalize=True) * 100
        # Setup plot
        fig1 = plt.figure(figsize=(8, 8))
        # Make the pie chart
        plt.pie(dist, labels=dist.index, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
        plt.title('EngagementLevel Distribution')
        # Show the pie chart
        st.pyplot(fig1)

        # Insight 1
        st.write('''The EngagementLevel classes are imbalanced, with 'Medium' dominating. To prevent model bias toward the 
                 majority class and ensure fair performance across all segments, the dataset is balanced using SMOTENC, 
                 which is suitable for mixed categorical and numerical features.''')

        # Plot 2
        st.subheader('2. Do People Who Play More Often Also Play Longer per Session?')
        st.write('''Next, we analyze the relationship between SessionsPerWeek and PlayTimeHours to see if 
                 more frequent sessions lead to longer total playtime. This is visualized using a scatterplot with a regression line.''')

        # Make a scatterplot with a regression line to highlight the correlation
        fig2 = sns.lmplot(data=df, x='SessionsPerWeek', y='PlayTimeHours', scatter_kws={}, line_kws={'color': 'firebrick'}, height=7, aspect=7/7)
        fig2.fig.suptitle('SessionsPerWeek VS PlayTimeHours', y=1.02)
        plt.grid(True)
        # Show the scatterplot
        st.pyplot(fig2.fig)

        # Insight 2
        st.write('''The scatterplot shows that SessionsPerWeek has little to no correlation with PlayTimeHours. 
                 The nearly flat regression line suggests players often have consistent total playtime regardless of 
                 how frequently they play—indicating shorter, more frequent sessions.''')

        # Plot 3
        st.subheader('3. Does Certain Genres Have Naturally Longer Play Sessions than Other Genres?')
        st.write('''We analyze the distribution of AvgSessionDurationMinutes across different GameGenre values using a boxplot. 
                 This helps identify which genres tend to have longer play sessions—an important insight for deciding where to focus development and resources.''')

        # Make a boxplot to highlight the average session distribution for each game genre
        fig3 = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='GameGenre', y='AvgSessionDurationMinutes')
        plt.title('GameGenre VS AvgSessionDuration')
        plt.grid(True)
        # Show the boxplot
        st.pyplot(fig3)

        # Insight 3
        st.write('''The boxplot shows that all GameGenre values have similar medians and IQRs, indicating that session duration doesn’t 
                 significantly vary by genre. This suggests that extended play sessions can happen across any genre, not just specific ones.''')

        # Plot 4
        st.subheader('4. How Does Game Difficulty Affects Player Engagement Across Genres?')
        st.write('''We examine the distribution of GameDifficulty across each GameGenre using a stacked barplot. This helps identify whether 
                 certain genres attract more casual or challenge-seeking players—valuable insight for deciding both the genre and difficulty level to focus on.''')

        # Make a stacked barplot to highlight the game difficulty distribution for each game genre
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        pd.crosstab(df['GameGenre'], df['GameDifficulty'], normalize='index').plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('GameGenre VS GameDifficulty')
        # Show the stacked barplot
        st.pyplot(fig4)

        # Insight 4
        st.write('''The plot shows that all GameGenres have a similar distribution of difficulty levels, with most players choosing Easy, 
                 followed by Medium and Hard. No genre stands out as attracting more challenge-seeking players—even genres like Strategy or RPG, which might be expected to.''')

        # Plot 5
        st.subheader('5. Do More Frequent Gamers Correlate with Them Having Higher Engagement Level?')
        st.write('''We group SessionsPerWeek into player types—Casual, Normal, Frequent, and Hardcore—to analyze how engagement levels vary by play frequency. 
                 This helps determine whether players who game more often also tend to be more engaged, using the stacked countplot below.''')

        # Make a stacked countplot to highlight the target's distribution against player types
        dfPlayerType = df.copy()
        # Bin the SessionsPerWeek column into four bins
        dfPlayerType['PlayerType'] = pd.cut(dfPlayerType['SessionsPerWeek'], bins=[-1, 2, 5, 10, 19], labels=['Casual', 'Normal', 'Frequent', 'Hardcore'])
        fig5 = plt.figure(figsize=(8, 5))
        sns.countplot(data=dfPlayerType, x='PlayerType', hue='EngagementLevel')
        plt.title('PlayerType (Custom Segments) VS EngagementLevel')
        # Show the stacked barplot
        st.pyplot(fig5)

        # Insight 5
        st.write('''The plot shows a clear pattern: Hardcore players mostly fall into the High engagement group, while Casual players tend to be Low engagement. 
                 Normal and Frequent players lean toward Medium, with some overlap into High and Low. This strong behavioral segmentation suggests that SessionsPerWeek is a valuable metric for targeted strategies, 
                 such as tailoring marketing messages by player type.''')

        # Plot 6
        st.subheader('6. Does Spending More Time In-Game Correlate with Paying More?')
        st.write('''We bin AvgSessionDurationMinutes into six ranges (spaced by 30 minutes) to better visualize its relationship with InGamePurchases. 
                 This helps reveal whether players with longer session durations are more likely to spend in-game, using the stacked countplot below.''')

        # Make a stacked countplot to highlight the session duration's distribution against in-game purchases
        dfSession = df.copy()
        # Bin the AvgSessionDurationMinutes column into six bins
        dfSession['SessionBin'] = pd.cut(dfSession['AvgSessionDurationMinutes'], bins=[10, 40, 70, 100, 130, 160, 180], labels=['10-40', '40-70', '70-100', '100-130', '130-160', '160-179'])
        fig6 = plt.figure(figsize=(8, 5))
        sns.countplot(data=dfSession, x='SessionBin', hue='InGamePurchases')
        plt.title('SessionLength (Custom Segments) VS InGamePurchases')
        # Show the stacked barplot
        st.pyplot(fig6)

        # Insight 6
        st.write('''The plot shows that while non-purchasing players dominate across all session length bins, purchasing players are consistently present. 
                 The purchase rate remains fairly stable, suggesting no strong link between session duration and likelihood of in-game purchases.''')

        # Plot 7
        st.subheader('7. Do Gamers That Grinds for Levels and Achievements Correlate with Them Having Higher Engagement Level?')
        st.write('''We use a heatmap to explore how EngagementLevel varies across binned PlayerLevel and AchievementsUnlocked. 
                 Although EngagementLevel is categorical, we use its encoded average here as a heuristic to reveal patterns across combined player traits—helping 
                 identify trends that may not be obvious when analyzed separately.''')

        # Make a heatmap to highlight the target's distribution against both player level and achievements unlocked
        dfLvlAch = df.copy()
        # Bin the PlayerLevel and AchievementsUnlocked columns into five bins
        dfLvlAch['LevelBin'] = pd.cut(dfLvlAch['PlayerLevel'], bins=5)
        dfLvlAch['AchievementBin'] = pd.cut(dfLvlAch['AchievementsUnlocked'], bins=5)
        # Encode and mean the target only for this heatmap as a heuristic for the plot
        dfLvlAch['EngagementLevel_enc'] = OrdinalEncoder(categories=[['Low', 'Medium', 'High']]).fit_transform(dfLvlAch[['EngagementLevel']])
        heatmap_data = dfLvlAch.groupby(['LevelBin', 'AchievementBin'])['EngagementLevel_enc'].mean().unstack()
        fig7 = plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
        plt.title('Average EngagementLevel by Level & Achievements')
        plt.xlabel('Achievements Unlocked')
        plt.ylabel('Player Level')
        # Show the heatmap
        st.pyplot(fig7)

        # Insight 7
        st.write('''The heatmap shows that Engagement Level increases with both PlayerLevel and AchievementsUnlocked. Players with high levels and many achievements are more likely to be highly engaged, 
                 while newer or less accomplished players tend to have lower engagement. This suggests strong potential for targeted strategies based on player progression.''')

        # Visualization with exploratory data analysis
        st.header("Overal Insights and Recomendations")
        st.markdown('''1. Segment Users by Engagement Style, Not Just Time
- Session frequency and duration tell different stories—some play often, others play long. Personalize based on style.
2. Focus Monetization on Behavior, Not Duration
- Purchase rates are consistent across session lengths. Prioritize smart in-game promotions over playtime targeting.
3. Frequent Players Drive ROI
- Hardcore and frequent users are the most engaged. Identify them early to offer exclusive deals and loyalty perks.
4. Genre & Difficulty Balance is Healthy so Try to Maintain It
- No genre skews heavily in difficulty. Maintain this balance with strong onboarding for all skill levels.
5. Don’t Overlook Casuals, Test Retention Features
- Casual players are numerous. A/B test features to improve their retention and gradually boost engagement.''')

if __name__ == '__main__':
        run()
