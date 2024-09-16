export type MovieTitle = {
  id: number;
  title: string;
};

export type MovieData = MovieTitle & {
  tmdb: string;
  img: string;
  description: string;
  tags: string[];
  release: string;
};

export type MovieList = {
  status: string;
  list: MovieData[];
};
