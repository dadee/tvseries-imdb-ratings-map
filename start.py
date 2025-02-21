import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path('data')
OUTPUT_DIR = Path('res_img')
DATABASE_PATH = Path('imdb_data.db')
OUTPUT_DIR.mkdir(exist_ok=True)

EPISODE_PATH = DATA_DIR / 'title.episode.tsv.gz'
RATINGS_PATH = DATA_DIR / 'title.ratings.tsv.gz'
BASICS_PATH = DATA_DIR / 'title.basics.tsv.gz'


def clean_text(text: str) -> str:
    """Remove special characters and extra spaces from text, keeping only letters, numbers, and single spaces."""
    if pd.isna(text):
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned = ' '.join(cleaned.split())
    return cleaned.lower()


def load_data_to_sqlite():
    """Load data into SQLite database, filtering out series with < 100 votes and removing titleType."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS episodes (
                        tconst TEXT PRIMARY KEY,
                        parentTconst TEXT,
                        seasonNumber INTEGER,
                        episodeNumber INTEGER)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS ratings (
                        tconst TEXT PRIMARY KEY,
                        averageRating REAL,
                        numVotes INTEGER)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS basics (
                        tconst TEXT PRIMARY KEY,
                        primaryTitle TEXT,
                        originalTitle TEXT,
                        startYear INTEGER,
                        cleaned_title TEXT)''')

    # Загрузка и фильтрация ratings
    cursor.execute("SELECT COUNT(*) FROM ratings")
    if cursor.fetchone()[0] == 0:
        print("Loading Ratings Data into SQLite with vote threshold >= 100...")
        rating_cols = ['tconst', 'averageRating', 'numVotes']
        df_ratings = pd.read_csv(
            RATINGS_PATH,
            compression='gzip',
            sep='\t',
            header=0,
            names=rating_cols,
            engine='python',
            na_values='\\N',
            on_bad_lines='skip'
        )
        df_ratings = df_ratings[df_ratings['numVotes'] >= 100]
        df_ratings.to_sql('ratings', conn, if_exists='replace', index=False)
        print("Creating index on tconst in ratings...")
        cursor.execute("CREATE INDEX idx_tconst_ratings ON ratings (tconst)")
        print("Ratings data and index loaded into database (filtered to >= 100 votes).")

    # Загрузка basics с фильтрацией по ratings и удалением titleType
    cursor.execute("SELECT COUNT(*) FROM basics")
    if cursor.fetchone()[0] == 0:
        print("Loading Basics Data into SQLite with vote threshold >= 100...")
        basics_cols = ['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'startYear']
        try:
            df_basics = pd.read_csv(
                BASICS_PATH,
                compression='gzip',
                sep='\t',
                header=0,
                names=basics_cols,
                engine='python',
                na_values='\\N',
                usecols=[0, 1, 2, 3, 5],
                on_bad_lines='skip'
            )
            df_basics = df_basics[df_basics['titleType'] == 'tvSeries']
            df_ratings = pd.read_sql_query("SELECT tconst FROM ratings", conn)
            df_basics = df_basics.merge(df_ratings, on='tconst', how='inner')
            df_basics['cleaned_title'] = df_basics['primaryTitle'].apply(clean_text)
            df_basics = df_basics.drop(columns=['titleType'])
            df_basics.to_sql('basics', conn, if_exists='replace', index=False)
            print("Basics data loaded into database (filtered to series with >= 100 votes, titleType removed).")
        except Exception as e:
            print(f"Ошибка при загрузке basics: {str(e)}")
            raise

    # Загрузка и фильтрация episodes
    cursor.execute("SELECT COUNT(*) FROM episodes")
    if cursor.fetchone()[0] == 0:
        print("Loading Episodes Data into SQLite with vote threshold >= 100...")
        episode_cols = ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber']
        df_episodes = pd.read_csv(
            EPISODE_PATH,
            compression='gzip',
            sep='\t',
            header=0,
            names=episode_cols,
            engine='python',
            na_values='\\N',
            on_bad_lines='skip'
        )
        df_basics = pd.read_sql_query("SELECT tconst FROM basics", conn)
        df_episodes = df_episodes[df_episodes['parentTconst'].isin(df_basics['tconst'])]
        df_episodes.to_sql('episodes', conn, if_exists='replace', index=False)
        print("Creating index on parentTconst...")
        cursor.execute("CREATE INDEX idx_parentTconst ON episodes (parentTconst)")
        print("Episodes data and index loaded into database (filtered to series with >= 100 votes).")
        print("Compacting database...")
        cursor.execute("VACUUM")
        print("Database compacted.")

    conn.commit()
    conn.close()


def get_series_id_from_title() -> tuple[str, str, str]:
    """Ask user for series title and find its ID in the database, sorted by popularity (numVotes)."""
    conn = sqlite3.connect(DATABASE_PATH)

    series_title = input("Введите название сериала (или 'exit' для выхода): ").strip()
    if series_title.lower() == 'exit':
        raise SystemExit("Программа завершена пользователем.")

    cleaned_search = clean_text(series_title)
    if not cleaned_search:
        conn.close()
        raise ValueError("Введите корректное название сериала.")
    search_term = f" {cleaned_search} "

    query = """
        SELECT b.tconst, b.primaryTitle, b.startYear, b.cleaned_title, COALESCE(r.numVotes, 0) as num_votes
        FROM basics b
        LEFT JOIN ratings r ON b.tconst = r.tconst
        WHERE (' ' || b.cleaned_title || ' ' LIKE ?)
    """
    cursor = conn.cursor()
    cursor.execute(query, (f"%{search_term}%",))
    results = cursor.fetchall()

    if not results:
        conn.close()
        raise ValueError(f"Сериал с названием '{series_title}' не найден в базе данных.")

    if len(results) == 1:
        series_id, found_title, start_year, cleaned_title, _ = results[0]
        year_str = f" ({int(start_year)})" if pd.notna(start_year) else " (год неизвестен)"
        print(f"Найден сериал: {found_title}{year_str} (ID: {series_id})")
        conn.close()
        return series_id, found_title, cleaned_title

    sorted_results = sorted(results, key=lambda x: x[4], reverse=True)[:10]

    print(f"Найдено несколько сериалов с похожим названием '{series_title}' (показаны первые 10 по популярности):")
    for i, (tconst, title, start_year, cleaned_title, num_votes) in enumerate(sorted_results, 1):
        year_str = f" ({int(start_year)})" if pd.notna(start_year) else " (год неизвестен)"
        print(f"{i}. {title}{year_str} (ID: {tconst}, Голосов: {num_votes})")

    while True:
        try:
            choice = int(input("Выберите номер сериала (1-{}): ".format(len(sorted_results))))
            if 1 <= choice <= len(sorted_results):
                series_id, found_title, start_year, cleaned_title, _ = sorted_results[choice - 1]
                year_str = f" ({int(start_year)})" if pd.notna(start_year) else " (год неизвестен)"
                print(f"Выбран сериал: {found_title}{year_str} (ID: {series_id})")
                conn.close()
                return series_id, found_title, cleaned_title
            else:
                print("Пожалуйста, введите число в диапазоне 1-{}.".format(len(sorted_results)))
        except ValueError:
            print("Пожалуйста, введите корректное число.")


def process_episode_data(series_id: str) -> pd.DataFrame:
    """Process episode data from SQLite database for given series ID."""
    conn = sqlite3.connect(DATABASE_PATH)

    query = """
        SELECT e.seasonNumber, e.episodeNumber, r.averageRating
        FROM episodes e
        LEFT JOIN ratings r ON e.tconst = r.tconst
        WHERE e.parentTconst = ? 
        AND e.seasonNumber IS NOT NULL 
        AND e.episodeNumber IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=(series_id,))
    conn.close()

    if df.empty:
        raise ValueError(f"No episodes found for series ID: {series_id}")

    df_final = (df.astype({'seasonNumber': int, 'episodeNumber': int, 'averageRating': float})
                .sort_values(['seasonNumber', 'episodeNumber'])
                .pivot(index='episodeNumber', columns='seasonNumber', values='averageRating'))

    return df_final


def create_visualization(df: pd.DataFrame, series_title: str, cleaned_title: str) -> None:
    """Create and save the visualization with 's' prefixed to season numbers and underscores in filename."""
    display_values = df.fillna('').apply(
        lambda x: x.map(lambda y: str(round(y, 1)) if isinstance(y, (int, float)) else y)
    )

    num_rows = len(df.index)
    num_cols = len(df.columns)

    base_width_per_col = 0.5
    base_height_per_row = 0.4
    min_width = 6
    min_height = 4

    fig_width = max(min_width, num_cols * base_width_per_col)
    fig_height = max(min_height, num_rows * base_height_per_row)

    norm = plt.Normalize(2, 10)
    colours = plt.cm.RdYlGn(norm(df.values))
    colours[np.isnan(df.values)] = [1, 1, 1, 1]

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    col_width = min(0.05, 1.0 / num_cols)

    col_labels = [f"s{col}" for col in df.columns]

    table = plt.table(
        cellText=display_values.values,
        rowLabels=df.index,
        colLabels=col_labels,
        colWidths=[col_width] * num_cols,
        cellLoc='center',
        loc='center',
        cellColours=colours
    )
    table.scale(1, 1.2)

    title_y = 0.95 - (num_rows * 0.005)
    title_y = max(0.85, title_y)
    plt.title(f"{series_title} Episode Ratings", pad=10, y=title_y)

    top_margin = max(0.85, 0.95 - (num_rows * 0.005))
    plt.subplots_adjust(top=top_margin, bottom=0.1, left=0.1, right=0.9)

    plt.figtext(0.5, 0.02, "@botsuperbot", ha="center", fontsize=8, color="gray")

    # Заменяем пробелы на подчеркивания в имени файла
    output_path = OUTPUT_DIR / f"{cleaned_title.replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {output_path}")


def main():
    """Main execution function with retry on series not found."""
    try:
        load_data_to_sqlite()
        while True:
            try:
                series_id, series_title, cleaned_title = get_series_id_from_title()
                df = process_episode_data(series_id)
                create_visualization(df, series_title, cleaned_title)
                break
            except ValueError as e:
                print(f"Ошибка: {str(e)}")
                print("Попробуйте ввести название снова.")
            except SystemExit:
                print("До свидания!")
                break
            except Exception as e:
                print(f"Произошла непредвиденная ошибка: {str(e)}")
                raise
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main()
