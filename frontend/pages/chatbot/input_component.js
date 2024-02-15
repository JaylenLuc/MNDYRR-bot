import TextField from "@mui/material/TextField";
import styles from './chat.module.css';
import Button from '@mui/material/Button';
import { useState } from "react";
const SearchBar = ({handleClick,searchQuery,setSearchQuery,btnDisabled, setBtnDisabled}) => (

    
    <form>
      <TextField
        id="search-bar"
        className={styles.searchbar}
        onInput={(e) => {
          setSearchQuery(e.target.value);
        }}
        onChange={(text) => setBtnDisabled(!text.target.value)}
        label="Enter what is on your mind"
        variant="filled"
        placeholder="Ask questions!"
        size="medium"
      />
      <Button variant="contained" disabled={btnDisabled} onClick={() => handleClick()}>Ask</Button> 
      
    </form>
);

export default SearchBar