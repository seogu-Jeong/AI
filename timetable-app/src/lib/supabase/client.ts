import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export const eventService = {
  async getEvents(userId: string) {
    const { data, error } = await supabase
      .from('events')
      .select('*')
      .eq('user_id', userId);
    if (error) throw error;
    return data;
  },
  
  async addEvent(event: any) {
    const { data, error } = await supabase
      .from('events')
      .insert([event]);
    if (error) throw error;
    return data;
  }
};
