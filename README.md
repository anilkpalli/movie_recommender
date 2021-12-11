# movie_recommender
 
This is a demo app for movie recommender systems.


## Recommender systems

The app has two recommender systems: 
1. By Genre
2. By Rating

Each of those systems can be explored by using respective tabs as shown below:

![image](https://user-images.githubusercontent.com/67958934/145664881-477c5fd1-e656-4838-b67c-4143fc3c4720.png)

### System-1: By Genre

Recommendation "By Genre" allows user to select a Genre and Sort_by options (Rating/Popularity) via dropdown feature.
User can also filter movies by the year of release date using a slider as shown below:

![image](https://user-images.githubusercontent.com/67958934/145664874-9e81a9a2-4252-4d41-b13f-8f2d960468da.png)

The recommended movies (Max: Top-5) are shown in table format along with respective images.
The "# Ratings" column specifies the sorting order by Popularity and "MR Rating" (Movie-Recommender rating) column specifies the sorting order by Rating (Weighted)

![image](https://user-images.githubusercontent.com/67958934/145664455-f22c4d2b-3b3b-4f8a-baae-e46f40320ae8.png)

### System-2: By Rating

Recommendation "By Rating" allows user to provide ratings for the already seen/known movies and based on user ratings the system would recommend movies that user might like

The layout is as shown below: The layout is split into 2-sections (LHS and RHS)

![image](https://user-images.githubusercontent.com/67958934/145664575-467e8039-19b5-446a-b448-7c038d8fe834.png)

* The LHS (User input) section:
   * Allows the user to browse the movies in a table and rate the movies on a scale of 1-5. The unrated movies can be left as 0
 ![image](https://user-images.githubusercontent.com/67958934/145664687-31f94f74-e5fe-4b8e-adcf-19d9b532e5a7.png)
   * User can also filter the movies using the filter section just below the table header across all the columns. By default the search is case-sensitive, however it can be disabled as shown by the tooltip when hover over the filter section
 ![image](https://user-images.githubusercontent.com/67958934/145664710-8668d22e-a65c-43c8-9937-5383574056cc.png)
* The RHS (Recommendation) section:
   * Displays the Top-9 recommended movies (Limited to 9 for better aesthetics)
   * Rate as many movies as possible for better recommendations
   * If no movies are rated by the user or the system could not generate enough recommendations then the remaining slots would be occupied by the most popular movies
   * The recommendations are based on IBCF. For efficiency purposes the IBCF similarity amtrix is pre-computed and stored in the data folder. The recommendations are computed using matrix products accordingly.
   * The recommendations are shown as movie posters. Very few movies do not have images available in which case the reommendation might not be returned in that particular slot

   Note that the recommendations are reactive for every user input and could have 1-2 secs lag 
