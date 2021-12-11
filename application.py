import dash
from dash import dash_table
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
from dash.dependencies import Output, Input, State
from plotly.subplots import make_subplots
import math
import re
import random

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, Dash, Trigger, FileSystemCache
import dash_dangerously_set_inner_html

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
# EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server

myurl = "https://liangfgithub.github.io/MovieData/"
ratings = pd.read_csv(myurl + 'ratings.dat?raw=true', sep="::", header=None)
ratings.columns =['UserID', 'MovieID', 'Rating', 'Timestamp']

movies = pd.read_csv(myurl + 'movies.dat?raw=true', sep="::", header=None, encoding='latin-1')
movies.columns = ['MovieID', 'Title', 'Genres']
movies["Year"] = movies["Title"].apply(lambda x: int(x[len(x)-5:len(x)-1]))
movies["Title"] = movies["Title"].apply(lambda x: x[:len(x)-7])

rtg_cnts = ratings.groupby("MovieID")["Rating"].aggregate([("# Ratings","count")])
all_movies_rtg = movies.merge(rtg_cnts, on="MovieID", how="inner")
all_movies_rtg = all_movies_rtg.sort_values(by="# Ratings", ascending=True)[["MovieID", "Title", 'Genres', "Year"]]
all_movies_rtg["My_Rating"] = 0
all_movies_rtg["S.No"] = pd.Series(range(1, len(all_movies_rtg) + 1)).values

genres = movies["Genres"].values.tolist()
genres_sublists = [i.split("|") for i in genres]
genres_all = [j for i in genres_sublists for j in i]
genres_unq = list(set(genres_all))
genres_unq = ["All"] + genres_unq

small_image_url = "https://liangfgithub.github.io/MovieImages/"

ibcf_sim = pd.read_csv("data/ibcf_sim.csv", index_col=0)


def recommend_movies_sys1(genre="All", sort_by="Rating", top_n=5, min_year=1900, max_year=2000):
    movies_yor = movies.loc[(movies["Year"] >= min_year) & (movies["Year"] <= max_year)]

    if genre == "All":
        movies_genre = movies_yor.copy()
    else:
        movies_genre = movies_yor.loc[movies_yor['Genres'].isin([genre]), :]

    ratings_genre = ratings.loc[ratings['MovieID'].isin(movies_genre["MovieID"]), :]

    min_votes = int(0.02 * ratings_genre["UserID"].nunique())
    mean_rtg = ratings_genre["Rating"].mean()
    ratings_genre_agg = ratings_genre.groupby("MovieID")["Rating"]\
                                     .aggregate([("# Ratings","count"),
                                                ("Avg Rating","mean"),
                                                ("Wtd Rating",lambda x: round(((x.count() / (x.count()+min_votes)) * x.mean()) + ((min_votes / (x.count()+min_votes)) * mean_rtg),1))]).reset_index()

    movie_genre_ratings = ratings_genre_agg.merge(movies_genre, on="MovieID", how="inner")

    movie_genre_ratings_imgs = movie_genre_ratings.copy()
    movie_genre_ratings_imgs["Image"] = movie_genre_ratings_imgs["MovieID"].apply(
        lambda x: '<img src="' + small_image_url + str(x) + '.jpg?raw=true" width="100" height="100"></img>')

    movie_genre_ratings_imgs_sort = movie_genre_ratings_imgs.sort_values(by="Avg Rating", ascending=False).copy()
    if sort_by.lower() == "rating":
        movie_genre_ratings_imgs_sort = movie_genre_ratings_imgs.sort_values(by="Wtd Rating", ascending=False).copy()
    elif sort_by.lower() == "popularity":
        movie_genre_ratings_imgs_sort = movie_genre_ratings_imgs.sort_values(by="# Ratings", ascending=False).copy()

    if len(movie_genre_ratings_imgs_sort) > top_n:
        movie_genre_ratings_imgs_sort_n = movie_genre_ratings_imgs_sort.iloc[:top_n, :].copy()
    else:
        movie_genre_ratings_imgs_sort_n = movie_genre_ratings_imgs_sort.copy()

    html_tbl = movie_genre_ratings_imgs_sort_n[["Image", "Title", "Year", "# Ratings", "Wtd Rating"]].to_html(
        escape=False, index=False, render_links=True, border=1)

    return (html_tbl)

def make_genre_options():
    ret = []
    for value in genres_unq:
        ret.append({"label": value, "value": value})
    return ret

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Movie Recommender", className="ml-2")
                    ),
                ],
                align="center",
            )
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

reco_tabs = html.Div([
    dcc.Tabs(id="tabs", value='genre',
        children=[
        dcc.Tab(label='By Genre', value='genre', style={"fontWeight":"bold"}),
        dcc.Tab(label='By Choice', value='choice', style={"fontWeight":"bold"}),
        ]),
    html.Div(id='tabs-output')
])

BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(reco_tabs, width=12),
            ],
            style={"marginTop": 10},
        ),
    ],
    fluid=True,
)

app.layout = html.Div(children=[NAVBAR, BODY])

def create_card(i):
    return dbc.Card(
                    [
                        dbc.CardImg(src=small_image_url + str(i) + '.jpg?raw=true', top=True, style={"maxHeight": "200px"}),
                    ],
                    style={"width": "18rem"},
                )

def create_layout(movie_ids=[1,2,3,4,5,6,7,8,9],nrow=3):
    grid = []
    for j in range(1,nrow+1):
         # print(j)
         grid.append(dbc.Row([
                        dbc.Col(create_card((movie_ids[((j-1)*3)+0]))),
                        dbc.Col(create_card((movie_ids[((j-1)*3)+1]))),
                        dbc.Col(create_card((movie_ids[((j-1)*3)+2]))),
                    ], style={"marginTop": 10}))
    return grid

from collections import OrderedDict
df = pd.DataFrame(OrderedDict([
    ('climate', ['Sunny', 'Snowy', 'Sunny', 'Rainy']),
    ('temperature', [13, 43, 50, 30]),
    ('city', ['NYC', 'Montreal', 'Miami', 'NYC'])
]))


@app.callback(Output('tabs-output','children'),
              [Input('tabs', 'value')])
def render_selections(input_value):
    if input_value=="genre":
        return html.Div([
                        dbc.Row([
                                dbc.Col(html.H5("Select a Genre", style={"marginTop": 20}), width=6),
                                dbc.Col(html.H5("Sort By", style={"marginTop": 20}), width=6),
                            ]),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id="genre-drop", options=make_genre_options(), value="All", clearable=False, style={"marginBottom": 5, "font-size": 14}, ),width=6),
                            dbc.Col(dcc.Dropdown(id="sort-by", options=[
                                        {'label': 'Top Rated', 'value': "Rating"},
                                        {'label': 'Most Popular', 'value': "Popularity"},
                                    ],
                                    value="Rating", style={"marginBottom": 5, "font-size": 14}, ),width=6),
                        ]),
                        html.H5("Year of Release", style={"marginTop": 20}),
                        dcc.RangeSlider(
                            id='yr-range-slider',
                            min=1910, max=2000, step=5,
                            value=[1910, 2000],
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        dcc.Loading([html.Div(id='genre-movies', style={"marginTop": 20})],type="dot", style={"marginTop":400}),
                        ], style={"width": '100%'})
    elif input_value=="choice":
        return html.Div(
                        html.Div([
                            html.Div(
                                [
                                    html.Div([html.H5("Rate as many movies as possible"), dash_table.DataTable(
                                            id='table-ratings',
                                            data=all_movies_rtg[["S.No","Title", "Year", "Genres", "My_Rating"]].to_dict('records'),
                                            columns=[
                                                {'id': 'S.No', 'name': 'S.No', 'type': 'numeric'},
                                                {'id': 'Title', 'name': 'Title', 'type': 'text'},
                                                {'id': 'Year', 'name': 'Year', 'type': 'numeric'},
                                                {'id': 'Genres', 'name': 'Genres', 'type': 'text'},
                                                {'id': 'My_Rating', 'name': 'My Rating (1-5)', 'type': 'numeric', 'presentation': 'dropdown'}
                                            ],
                                            filter_action='native',
                                            css=[{
                                                'selector': 'table',
                                                'rule': 'table-layout: fixed'  # note - this does not work with fixed_rows
                                            }],
                                            editable=True,
                                            # clearable=False,
                                            dropdown={
                                                'My_Rating': {
                                                    'options': [
                                                        {'label': i, 'value': i}
                                                        for i in [0,1,2,3,4,5]
                                                    ],'clearable': False
                                                },
                                            },
                                            # page_action='none',
                                            page_size=15,
                                            # fixed_rows={'headers': True},
                                            style_cell={'textAlign': 'left', 'fontSize': 15,},
                                            style_data={
                                                'color': 'black',
                                                'backgroundColor': 'white',
                                                'whiteSpace': 'normal',
                                                'height': 'auto',
                                            },
                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': 'rgb(220, 220, 220)',
                                                },
                                                {'if': {'column_id': 'S.No'},
                                                 'width': '5%'},
                                            ],
                                            style_header={
                                                'backgroundColor': "#1E90FF",
                                                'fontSize': 15,
                                                # 'fontWeight': 'bold',
                                                'textAlign': 'center',
                                            },
                                            style_table={'maxHeight':'600px','overflowY': 'auto'}
                                    ),], className="six columns"),
                                    html.Div(id='recom-movies',
                                             style={"maxHeight": "700px",},
                                             className="six columns"),
                                ],className="row"
                            )], style={"marginTop": 20}
                        )
                    )

@app.callback(Output('genre-movies', 'children'),
              [Input("genre-drop", "value"),
               Input("sort-by", "value"),
               Input('yr-range-slider', 'value')])
def populate_genres(gnr_value, srt_value, yr_value):

    html_tbl = recommend_movies_sys1(genre = gnr_value, sort_by=srt_value, top_n=5, min_year=yr_value[0], max_year=yr_value[1])
    html_tbl = re.sub(r'<table([^>]*)>', r'<table\1 width=100% >', html_tbl)

    return dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_tbl)

@app.callback(Output('recom-movies', 'children'),
                [Input("table-ratings", "data")])
def recommend_movies(rtgs_list):

    input_df = pd.DataFrame(rtgs_list)

    rtgs_df = all_movies_rtg.copy()

    if input_df["My_Rating"].sum()>0:

        rtgs_df["My_Rating"] = input_df["My_Rating"].values.tolist()
        rtgs_df["tmp"] = rtgs_df["MovieID"].apply(lambda x: "m" + str(x))
        temp = rtgs_df[["tmp", "My_Rating"]]
        zv = temp[temp["My_Rating"] == 0]["tmp"].values.tolist()

        temp_srt = temp.sort_values(by="tmp").T.reset_index(drop=True)
        temp_srt.columns = temp_srt.iloc[0]
        temp_srt = temp_srt[1:].copy()

        iv = ibcf_sim.values
        tv = temp_srt.values

        ivi = iv
        ivi[ivi != 0] = 1
        tvi = tv
        tvi[tvi != 0] = 1

        nmr = np.matmul(iv, tv.T)
        dmr = np.matmul(ivi, tvi.T)
        new_rtgs = np.divide(nmr, dmr, out=np.zeros_like(nmr), where=dmr != 0)
        new_rtgs_srs = pd.Series([new_rtgs[i][0] for i in range(new_rtgs.shape[0])], index=ibcf_sim.columns)
        new_rtgs_srs_zv = new_rtgs_srs.filter(items=zv, axis=0)
        srt_ids = new_rtgs_srs_zv.nlargest(9).index.tolist()

        ids = [int(i[1:]) for i in srt_ids][:9]

    else:

        ids = rtgs_df["MovieID"][:9].values.tolist()

    return [html.H5("Recommended Movies")] + create_layout(movie_ids=ids)

if __name__ == "__main__":
    application.run(debug=True)
