import { MovieData } from "@src/store/type";
import { CircularProgress } from "@mui/material";
import MovieCard from "./MovieCard";
import { css } from "@emotion/react";

export default function MovieList(props: { status: string; movieList: MovieData[] }) {
  return (
    <>
      {props.status === "loading" ? (
        <div
          css={css`
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            width: 300px;
          `}
        >
          <CircularProgress />
        </div>
      ) : (
        <div
          css={css`
            display: flex;
            flex-direction: row;
            height: 300px;
            gap: 20px;
          `}
        >
          {props.movieList.map((movie) => (
            <MovieCard movie={movie} width={100} height={150} />
          ))}
        </div>
      )}
    </>
  );
}
