import { MovieData } from "@src/store/type";
import { css } from "@emotion/react";
import { Typography } from "@mui/material";
import { useState } from "react";

function MovieTitle(props: { width: number; height: number; movie: MovieData }) {
  const titleHeight = 50;
  const tailHeight = 15;
  const padding = 10;
  return (
    <div
      css={css`
        padding: ${padding}px;
        display: flex;
        flex-direction: column;
        border-radius: 0px 10px 10px 0px;
        background-color: #323232;
        height: ${props.height - padding * 2}px;
      `}
    >
      <div
        css={css`
          display: flex;
          flex-direction: row;
          height: ${titleHeight}px;
          align-items: flex-end;
          justify-content: space-between;
        `}
      >
        <Typography variant="h4">{props.movie.title}</Typography>
        <Typography variant="subtitle2">{props.movie.release}</Typography>
      </div>
      <div
        css={css`
          padding-top: ${padding}px;
          height: ${props.height - titleHeight - tailHeight - padding * 3}px;
          width: ${props.width + 300}px;
          overflow: auto;
        `}
      >
        <Typography variant="body2">{props.movie.description}</Typography>
      </div>
      <div
        css={css`
          height: ${tailHeight}px;
          display: flex;
          flex-direction: row;
          justify-content: flex-end;
        `}
      >
        <Typography variant="caption">
          {props.movie.tags.reduce((a, c, i) => {
            if (i == 0) {
              return a + c;
            } else {
              return a + ", " + c;
            }
          }, "")}
        </Typography>
      </div>
    </div>
  );
}

export default function MovieCard(props: { movie: MovieData; width: number; height: number }) {
  const [hover, setHover] = useState(false);
  return (
    <div
      css={css`
        display: flex;
        flex-direction: row;
        border-radius: 10px;
        transition: 0.8s ease;
      `}
      onMouseLeave={() => setHover(false)}
      onMouseEnter={() => setHover(true)}
    >
      <img
        src={props.movie.img}
        css={css`
          border-radius: 10px;
          width: ${hover ? props.width * 1.8 : props.width}px;
          height: ${hover ? props.height * 1.8 : props.height}px;
          ${hover && "border-radius: 10px 0px 0px 10px;"}
        `}
      />
      {hover && (
        <MovieTitle movie={props.movie} width={props.width * 1.8} height={props.height * 1.8} />
      )}
    </div>
  );
}
