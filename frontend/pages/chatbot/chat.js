import Head from 'next/head';
import styles from './chat.module.css';
import {Box,Button} from '@mui/material';
import { useState } from "react";
import SearchBar from "./input_component"
import { shadows } from '@mui/system';
import { borders } from '@mui/system';
import axios from 'axios';
export default function chat() {
  //temporary, this variable will store the last valid server return value, the 
  //final version must store all responses 
  const [_temp, _set_temp] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const [btnDisabled, setBtnDisabled] = useState(true);

  function handleClick(){
    axios.get('http://127.0.0.1:8000/ai/query',{ params: { question: searchQuery } })
    .then(res => {

      console.log(res['data']['response'])
      _set_temp(res['data']['response'])


    })
    .catch(err => { 
      console.log(err) 
    })

  }

    return (
      <div>
        <Head>
          <title>Main Page</title>
          <link rel="icon" href="/favicon.ico" />
        </Head>
  
        <main>
        <Box color="black" bgcolor="beige" className= {styles.mainbox}> 

        <SearchBar handleClick={handleClick} searchQuery={searchQuery} setSearchQuery={setSearchQuery} btnDisabled={btnDisabled} setBtnDisabled={setBtnDisabled} />  
        {
          _temp ? _temp : null
        }   
        </Box> 

        </main>
      </div>
    );


}