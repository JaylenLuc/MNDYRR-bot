import TextField from "@mui/material/TextField";
import styles from './components.module.css'
import Button from '@mui/material/Button';
import { useState } from "react";
const SearchBar = ({handleClick,searchQuery,setSearchQuery,btnDisabled, setBtnDisabled}) => (

    
    <form style={{zIndex : 0}}>
      <TextField style={{zIndex : "inherit"}}
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
      <Button onTap variant="contained" disabled={btnDisabled} onClick={() => handleClick()}>Ask</Button> 
      
    </form>
);

export default SearchBar