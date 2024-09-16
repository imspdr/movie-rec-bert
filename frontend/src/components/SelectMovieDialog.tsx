import { observer } from "mobx-react";
import { useRootStore } from "@src/store/RootStoreProvider";
import { css } from "@emotion/react";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
import { Button, Dialog, DialogTitle, DialogContent, DialogActions } from "@mui/material";
import ClearIcon from "@mui/icons-material/Clear";

function MovieCard(props: { title: string; onDelete: () => void }) {
  return (
    <div
      css={css`
        height: 20px;
        width: 300px;
        padding: 10px;
        border-radius: 10px;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        background-color: var(--mint);
      `}
    >
      {props.title}
      <ClearIcon onClick={props.onDelete} />
    </div>
  );
}

function SelectMovieDialog(props: { open: boolean; setOpen: (v: boolean) => void }) {
  const rootStore = useRootStore();
  const handleClose = () => {
    props.setOpen(false);
  };
  const handleComplete = () => {
    rootStore.buildList();
    props.setOpen(false);
  };
  return (
    <Dialog
      open={props.open}
      onClose={handleClose}
      css={css`
        & .MuiDialog-paper {
          height: 600px;
          width: 800px;
        }
      `}
      maxWidth={false}
    >
      <DialogTitle>{"영화 선택하기"}</DialogTitle>
      <DialogContent>
        <div
          css={css`
            padding: 10px;
            display: flex;
            flex-direction: row;
            justify-content: space-between;
          `}
        >
          <div
            css={css`
              display: flex;
              flex-direction: column;
            `}
          >
            <Autocomplete
              disablePortal
              options={rootStore.moviePool.map((movie) => {
                return {
                  label: movie.title,
                  id: String(movie.id),
                };
              })}
              sx={{ width: 300, height: 60 }}
              renderInput={(params) => <TextField {...params} label="영화 선택" />}
              onChange={(e, v) => {
                if (v && v.id) {
                  rootStore.addMovie({
                    title: v.label,
                    id: Number(v.id),
                  });
                }
              }}
              css={css`
                margin-bottom: 30px;
              `}
            />
            <span>재밌게 본 영화를 검색한 뒤 선택하여</span>
            <span>내 영화 목록에 추가하세요!</span>
          </div>
          <div
            css={css`
              display: flex;
              flex-direction: column;
              gap: 10px;
              align-items: center;
              overflow: auto;
              height: 400px;
              width: 340px;
              padding: 10px;
            `}
          >
            <span>내 영화 목록</span>
            {rootStore.myMovieList.map((movie) => {
              return (
                <MovieCard
                  title={movie.title}
                  onDelete={() => {
                    rootStore.deleteMovie(movie.id);
                  }}
                />
              );
            })}
          </div>
        </div>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>취소</Button>
        <Button onClick={handleComplete} color="primary" autoFocus>
          완료
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default observer(SelectMovieDialog);
