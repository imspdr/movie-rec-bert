import { css } from "@emotion/react";
import { observer } from "mobx-react";
import { useRootStore } from "@src/store/RootStoreProvider";

function App() {
  const rootStore = useRootStore();
  return (
    <div
      css={css`
        display: flex;
        width: 99vw;
        flex-direction: column;
        align-items: center;
      `}
    ></div>
  );
}

export default observer(App);
