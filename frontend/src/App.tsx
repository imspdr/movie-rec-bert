import { css } from "@emotion/react";
import { observer } from "mobx-react";
import { useRootStore } from "@src/store/RootStoreProvider";
import { useState } from "react";
import SelectMovieDialog from "./components/SelectMovieDialog";
import { Button } from "@mui/material";
import MovieList from "./components/MovieList";

function App() {
  const rootStore = useRootStore();
  const [open, setOpen] = useState(false);
  return (
    <div
      css={css`
        display: flex;
        width: 99vw;
        flex-direction: column;
        align-items: flex-start;
      `}
    >
      <Button
        onClick={() => {
          setOpen(true);
        }}
      >
        {"영화 목록 편집하기"}
      </Button>
      <MovieList
        status={rootStore.myMovieListBuilt.status}
        movieList={rootStore.myMovieListBuilt.list}
      />
      <Button
        onClick={() => {
          rootStore.buildResult();
        }}
      >
        {"영화 추천하기"}
      </Button>
      <MovieList
        status={rootStore.resultMovieList.status}
        movieList={rootStore.resultMovieList.list}
      />
      <SelectMovieDialog open={open} setOpen={setOpen} />
    </div>
  );
}

export default observer(App);
