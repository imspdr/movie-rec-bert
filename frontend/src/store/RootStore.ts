import { runInAction, makeAutoObservable } from "mobx";
import { MovieTitle, MovieData, MovieList } from "./type";
import moviePool from "./movie_pool.json";
import { recAPI } from "./apis";

export class RootStore {
  private _moviePool: MovieTitle[];
  private _movieCache: MovieData[];
  private _myMovieList: MovieTitle[];
  private _myMovieListBuilt: MovieList;
  private _resultMovieList: MovieList;

  constructor() {
    this._moviePool = moviePool;
    this._movieCache = [];
    this._myMovieList = [];
    this._myMovieListBuilt = {
      status: "ready",
      list: [],
    };
    this._resultMovieList = {
      status: "ready",
      list: [],
    };
    makeAutoObservable(this);
  }
  get moviePool() {
    return this._moviePool;
  }
  get movieCache() {
    return this._movieCache;
  }
  get myMovieList() {
    return this._myMovieList;
  }
  get myMovieListBuilt() {
    return this._myMovieListBuilt;
  }
  get resultMovieList() {
    return this._resultMovieList;
  }
  set movieCache(given: MovieData[]) {
    this._movieCache = given;
  }
  set myMovieList(given: MovieTitle[]) {
    this._myMovieList = given;
  }
  set myMovieListBuilt(given: MovieList) {
    this._myMovieListBuilt = given;
  }
  set resultMovieList(given: MovieList) {
    this._resultMovieList = given;
  }

  deleteMovie = (id: number) => {
    this.myMovieList = this.myMovieList.filter((movie) => movie.id !== id);
  };
  addMovie = (movie: MovieTitle) => {
    if (this.myMovieList.find((myMovie) => myMovie.id === movie.id)) {
      return;
    }
    this.myMovieList = [...this.myMovieList.slice(0, 99), movie];
  };
  buildList = async () => {
    this.myMovieListBuilt = {
      list: [],
      status: "loading",
    };
    for (let i = 0; i < this.myMovieList.length; i++) {
      let movie = this.myMovieList[i];
      if (movie) {
        if (this.myMovieListBuilt.list.find((listMovie) => listMovie.id === movie!.id)) continue;

        const cacheMovie = this.movieCache.find((cacheMovie) => cacheMovie.id === movie!.id);
        if (cacheMovie) {
          this.myMovieListBuilt = {
            ...this.myMovieListBuilt,
            list: [...this.myMovieListBuilt.list, cacheMovie],
          };
        } else {
          const movieData = await recAPI.getMovieInfo(movie.id);
          this.myMovieListBuilt = {
            ...this.myMovieListBuilt,
            list: [...this.myMovieListBuilt.list, ...movieData],
          };
        }
      }
    }

    this.myMovieListBuilt = {
      ...this.myMovieListBuilt,
      status: "ready",
    };
  };
  buildResult = async () => {
    this.resultMovieList = {
      list: [],
      status: "loading",
    };

    const res = await recAPI.getrecommendationData(
      this.myMovieList.map((movie) => movie.id),
      10
    );

    for (let i = 0; i < res.length; i++) {
      let movie = res[i];
      if (movie) {
        const cacheMovie = this.movieCache.find((cacheMovie) => cacheMovie.id === movie);
        if (cacheMovie) {
          this.resultMovieList = {
            ...this.resultMovieList,
            list: [...this.resultMovieList.list, cacheMovie],
          };
        } else {
          const movieData = await recAPI.getMovieInfo(movie);
          this.resultMovieList = {
            ...this.resultMovieList,
            list: [...this.resultMovieList.list, ...movieData],
          };
        }
      }
    }
    this.resultMovieList = {
      ...this.resultMovieList,
      status: "ready",
    };
  };
}
