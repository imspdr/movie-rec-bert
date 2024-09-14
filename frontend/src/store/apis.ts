import axios from "axios";

const recommandURL = "/api/v1/models/movie-rec-bert-serving:predict";
const recommandHost = "movie-rec-bert-serving.default.example.com";

const movieApiURL = "/api/v1/models/movie-rec-bert-api:predict";
const movieApiHost = "movie-rec-bert-api.default.example.com";

export const recAPI = {
  getRecommandationData: async (movieIds: number[], topk: number) => {
    const ret = await axios
      .post(
        recommandURL,
        {
          instances: movieIds,
          params: {
            topk: topk,
          },
        },
        {
          headers: {
            "Content-Type": "application/json",
            "Kserve-Host": recommandHost,
          },
        }
      )
      .then((data: any) => {
        return data.data;
      })
      .catch((e) => {
        return { predictions: [] };
      });
    //const ret = inputDataSample;
    return ret.predictions as number[];
  },
  getMovieInfo: async (movieId: number) => {
    const ret = await axios
      .post(
        movieApiURL,
        {
          instances: [movieId],
          params: {
            do_predict: 1,
            periods: 100,
          },
        },
        {
          headers: {
            "Content-Type": "application/json",
            "Kserve-Host": movieApiHost,
          },
        }
      )
      .then((data: any) => {
        return data.data;
      })
      .catch((e) => {
        return { predictions: [] };
      });
    //const ret = predictionSample;
    return ret.predictions as number[];
  },
};
