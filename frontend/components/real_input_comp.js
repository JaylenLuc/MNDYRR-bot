import TextField from "@mui/material/TextField";
import styles from './components.module.css'
import Button from '@mui/material/Button';
import { useState } from "react";
const RealSearchBar = ({handleClick,searchQuery,setSearchQuery,btnDisabled, setBtnDisabled}) => (
  // onInput={(e) => {
  //   setSearchQuery(e.target.value);
  // }}
  // onChange={(text) => setBtnDisabled(!text.target.value)}
    
    <form style={{zIndex : 0}} className={styles.querybar}>
      <textarea className={styles.textBar}
      
            role="textbox" 
            onInput={(e) => {
            setSearchQuery(e.target.value);
            }}
            onChange={(text) => setBtnDisabled(!text.target.value)}
            placeholder="Ask Mendy anything! What's on your mind?"
            contenteditable
            >
            
      </textarea>

      <Button className = {styles.textButton} onTap variant="contained" disabled={btnDisabled} onClick={() => handleClick()}>Ask</Button> 
      
    </form>
);

export default RealSearchBar;